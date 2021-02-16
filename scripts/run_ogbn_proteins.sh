if [ $# -lt 1 ]; then
    echo "Usage: run_ogbn_proteins <log_dir> [<gpu_id>] [--test]"
    exit -1
fi

LOG_DIR=$1
if [ ! -d $LOG_DIR ]; then
    echo "Directory $LOG_DIR does not exist!"
    exit -1
fi

if [ $# -lt 2 ]; then
    DEVICE="0"
else
    DEVICE=$2
fi

TEST=""
if [ $# -ge 3 ] && [ $3 == "--test" ]; then
    TEST="--test"
    echo "[Testing Mode]"
else
    echo "[Training Mode]"
fi

python main_proteins_nodeproppred.py --model gcn --device $DEVICE --runs 10 --epochs 1000 --hidden_channels 256 --filename "$LOG_DIR/proteins_gcn.log" $TEST
python main_proteins_nodeproppred.py --model gin --device $DEVICE --runs 10 --epochs 1000 --hidden_channels 200 --filename "$LOG_DIR/proteins_gin.log" $TEST
python main_proteins_nodeproppred.py --model sogcn --K 2 --device $DEVICE --runs 10 --epochs 3000 --hidden_channels 200 --filename "$LOG_DIR/proteins_sogcn_2.log" $TEST
python main_proteins_nodeproppred.py --model sogcn --K 4 --device $DEVICE --runs 10 --epochs 3000 --hidden_channels 200 --filename "$LOG_DIR/proteins_sogcn_4.log" $TEST
python main_proteins_nodeproppred.py --model sogcn --K 6 --device $DEVICE --runs 10 --epochs 3000 --hidden_channels 200 --filename "$LOG_DIR/proteins_sogcn_6.log" $TEST
python main_proteins_nodeproppred.py --model gcnii --device $DEVICE --runs 10 --epochs 3000 --hidden_channels 256 --filename "$LOG_DIR/proteins_gcnii.log" $TEST
python main_proteins_nodeproppred.py --model gcnii_star --device $DEVICE --runs 10 --epochs 3000 --hidden_channels 256 --filename "$LOG_DIR/proteins_gcnii_star.log" $TEST
python main_proteins_nodeproppred.py --model appnp --device $DEVICE --runs 10 --epochs 3000 --hidden_channels 256 --filename "$LOG_DIR/proteins_appnp.log" $TEST
python main_proteins_nodeproppred.py --model arma --device $DEVICE --runs 10 --epochs 1000 --hidden_channels 256 --filename "$LOG_DIR/proteins_arma.log" $TEST
python main_proteins_nodeproppred.py --model graphsage --device $DEVICE --runs 10 --epochs 1000 --hidden_channels 256 --filename "$LOG_DIR/proteins_graphsage.log" $TEST