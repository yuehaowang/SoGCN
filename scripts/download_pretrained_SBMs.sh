DIR=./_out/SBMs_node_classification
[ ! -d $DIR ] && mkdir -p $DIR
cd $DIR

for FILE in "oy42zsgb7x5d5s3/cluster_sogcn.zip" "973pag4timdsb8r/cluster_sogcn_gru.zip" "dyr83s19yu5naia/pattern_sogcn.zip" "z73nd2o6z48i4i8/pattern_sogcn_gru.zip"
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