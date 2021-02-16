DIR=./_out/molecules_graph_regression
[ ! -d $DIR ] && mkdir -p $DIR
cd $DIR

for FILE in "24un9ivqjjjdjnz/zinc_sogcn.zip" "703mcmtb8cg1isj/zinc_sogcn_gru.zip"
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