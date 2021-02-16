DIR=./_out/superpixels_graph_classification
[ ! -d $DIR ] && mkdir -p $DIR
cd $DIR

for FILE in "kccei4qjg4x1ycz/cifar10_sogcn.zip" "nkswz46y48ceorf/cifar10_sogcn_gru.zip" "0btwx3s3f60s7sd/mnist_sogcn.zip" "tzkt1t64o6041rr/mnist_sogcn_gru.zip"
do
    FILE_NAME="$(basename "${FILE}" ".${FILE##*.}")"
	if test -d "$FILE_NAME"; then
        echo -e "$FILE_NAME already exists."
    else
        echo -e "\ndownloading $FILE_NAME..."
        curl https://www.dropbox.com/s/$FILE?dl=1 -o $FILE_NAME.zip -J -L -k
        echo -e "unzipping..."
        unzip -q -n $FILE_NAME.zip
    fi
done