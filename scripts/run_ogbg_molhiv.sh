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

python main_molhiv_graphproppred.py --dataset ogbg-molhiv --gnn gcn --device $DEVICE --runs 4 --epochs 100 --filename "$LOG_DIR/molhiv_gcn.log" $TEST
python main_molhiv_graphproppred.py --dataset ogbg-molhiv --gnn gin --device $DEVICE --runs 4 --epochs 100 --filename "$LOG_DIR/molhiv_gin.log" $TEST
python main_molhiv_graphproppred.py --dataset ogbg-molhiv --gnn sogcn --K 2 --device $DEVICE --runs 4 --epochs 500 --filename "$LOG_DIR/molhiv_sogcn_2.log" $TEST
python main_molhiv_graphproppred.py --dataset ogbg-molhiv --gnn sogcn --K 4 --device $DEVICE --runs 4 --epochs 500 --filename "$LOG_DIR/molhiv_sogcn_4.log" $TEST
python main_molhiv_graphproppred.py --dataset ogbg-molhiv --gnn sogcn --K 6 --device $DEVICE --runs 4 --epochs 500 --filename "$LOG_DIR/molhiv_sogcn_6.log" $TEST
python main_molhiv_graphproppred.py --dataset ogbg-molhiv --gnn gcnii --device $DEVICE --runs 4 --epochs 500 --filename "$LOG_DIR/molhiv_gcnii.log" $TEST
python main_molhiv_graphproppred.py --dataset ogbg-molhiv --gnn appnp --device $DEVICE --runs 4 --epochs 100 --filename "$LOG_DIR/molhiv_appnp.log" $TEST
python main_molhiv_graphproppred.py --dataset ogbg-molhiv --gnn arma --device $DEVICE --runs 4 --epochs 500 --filename "$LOG_DIR/molhiv_arma.log" $TEST
python main_molhiv_graphproppred.py --dataset ogbg-molhiv --gnn graphsage --device $DEVICE --runs 4 --epochs 100 --filename "$LOG_DIR/molhiv_graphsage.log" $TEST
