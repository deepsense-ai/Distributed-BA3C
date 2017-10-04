#! /bin/bash
echo "SLURM_JOB_ID " $SLURM_JOB_ID  "; SLURM_JOB_NAME " $SLURM_JOB_NAME "; SLURM_JOB_NODELIST " $SLURM_JOB_NODELIST "; SLURMD_NODENAME " $SLURMD_NODENAME  "; SLURM_JOB_NUM_NODES " $SLURM_JOB_NUM_NODES

PORT=$1
TF_PORT=$2
ENVIRONMENT=$3
OPTIMIZER=$4
USE_SYNC_OPT=$5
EXPERIMENT_TAGS=$6
EXPERIMENT_NAME=$7
LEARNING_RATE=$8
BATCH_SIZE=$9
NUM_GRAD=${10}
INTEL_TF=${11}
EARLY=${12}
FC_NEURONS=${13}
SIMULATOR_PROCESSES=${14}
PS=${15}
FC_INIT=${16}
CONV_INIT=${17}
OFFLINE=${18}
EXPDIR=${19}
REPLACE_WITH_CONV=${20}
FC_SPLITS=${21}
DEBUG_CHARTS=${22}
LOG_DIR=${23}${EXPERIMENT_NAME}
EPSILON=${24}
BETA1=${25}
BETA2=${26}
SAVE=${27}
ADAM_DEBUG=${28}
SAVE_OUTPUT=${29}
EVAL_NODE=${30}
RECORD_NODE=${31}
RECORD_LENGTH=${32}
SCHEDULE_HYPER=${33}

################################################
# write your info here !!!
EXPERIMENTS_DIR=/net/people/plgtgrel/adam/experiments/
VIRTUAL_ENV=/net/people/plgtgrel/adam/env/test_env
DISTRIBUTED_A3C_PATH=/net/people/plgtgrel/adam/distributed_tp/
export TENSORPACK_PIPEDIR=/net/archive/groups/plggluna/intel_2/tmp_sockets/
################################################

EXPERIMENT_DIR="${EXPERIMENTS_DIR}/${EXPERIMENT_NAME}"
MODELS_DIR="${EXPERIMENT_DIR}/models/"
STORAGE="${EXPERIMENT_DIR}/storage/"
LOG_DIR="${EXPERIMENT_DIR}/logs/"

if [ ! -d "${EXPERIMENTS_DIR}/${EXPERIMENT_NAME}" ]; then
    mkdir $EXPERIMENT_DIR
    mkdir $MODELS_DIR
    mkdir $STORAGE

    if [ "$SAVE_OUTPUT" == "True" ]
    then
        mkdir $LOG_DIR
    fi
fi

TENSORPACK_CPU_PATH="${DISTRIBUTED_A3C_PATH}/src/tensorpack_cpu/"
OPENAI_GYM_PATH="${DISTRIBUTED_A3C_PATH}/src/OpenAIGym/"

module load plgrid/tools/python/2.7.13
module load plgrid/libs/mkl/2017.0.0
module load tools/gcc/6.2.0

source ${VIRTUAL_ENV}/bin/activate
export PYTHONPATH=${DISTRIBUTED_A3C_PATH}/src/tensorpack_cpu:$PYTHONPATH
export PYTHONPATH=${VIRTUAL_ENV}/bin/python:${VIRTUAL_ENV}/lib/python2.7/site-packages:$PYTHONPATH

is_chief="$(python is_chief.py ${OPENAI_GYM_PATH} ${PS} ${EVAL_NODE} ${RECORD_NODE})"

PROGRAM_PATH="${OPENAI_GYM_PATH}/train.py"
TAGS="$GIT_HASH $EXPERIMENT_TAGS"
NAME="$EXPERIMENT_NAME"
PROJECT=Distributed_A3C
COPY=${TENSORPACK_CPU_PATH}
PROGRAM_ARGS="--mkl 0 --dummy 0 --sync 0 --cpu 1 --artificial_slowdown 0 --queue_size 1 --my_sim_master_queue 1 --train_log_path ${STORAGE}/atari_trainlog/ --predict_batch_size 16 --dummy_predictor 0 --do_train 1 --simulator_procs $SIMULATOR_PROCESSES --env $ENVIRONMENT --nr_towers 1 --nr_predict_towers 3 --steps_per_epoch 1000 --fc_neurons $FC_NEURONS --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --port $PORT --tf_port $TF_PORT --optimizer $OPTIMIZER --use_sync_opt $USE_SYNC_OPT --num_grad $NUM_GRAD  --early_stopping $EARLY  --ps $PS  --fc_init $FC_INIT  --conv_init $CONV_INIT --replace_with_conv $REPLACE_WITH_CONV --fc_splits $FC_SPLITS --debug_charts $DEBUG_CHARTS --epsilon $EPSILON --beta1 $BETA1 --beta2 $BETA2 --save_every $SAVE --models_dir $MODELS_DIR --experiment_dir $EXPERIMENT_DIR --adam_debug $ADAM_DEBUG --eval_node $EVAL_NODE --record_node $RECORD_NODE --schedule_hyper $SCHEDULE_HYPER"

if [ "$is_chief" == "eval" ]
then
    echo "STARING EVAL NODE"
    python ${OPENAI_GYM_PATH}/eval_model.py --fc_neurons $FC_NEURONS --fc_splits $FC_SPLITS --models_dir $MODELS_DIR --server_port $PORT --env $ENVIRONMENT --replace_with_conv $REPLACE_WITH_CONV --ps $PS 2>&1 | tee -a ${LOG_DIR}/output_${SLURMD_NODENAME}.log
    exit
fi

if [ "$is_chief" == "record" ]
then
    echo "STARTING RECORD NODE"
    python ${OPENAI_GYM_PATH}/record_model.py --fc_neurons $FC_NEURONS --fc_splits $FC_SPLITS --env $ENVIRONMENT --replace_with_conv $REPLACE_WITH_CONV --models_dir $MODELS_DIR --time $RECORD_LENGTH 2>&1 | tee -a ${LOG_DIR}/output_${SLURMD_NODENAME}.log
    exit
fi


echo "PROGRAM_ARGS: " $PROGRAM_ARGS
echo "OFFLINE:" $OFFLINE

if [ "$SAVE_OUTPUT" == "True" ]
then
    python $PROGRAM_PATH $PROGRAM_ARGS 2>&1 | tee -a ${LOG_DIR}/output_${SLURMD_NODENAME}.log
else
    python $PROGRAM_PATH $PROGRAM_ARGS
fi
echo "DONE"
