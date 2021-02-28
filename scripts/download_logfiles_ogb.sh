DIR=./_out/OGB
[ ! -d $DIR ] && mkdir -p $DIR
cd $DIR

for FILE in "5al33zc0czruajl/ogbn_proteins.zip" "xtotwq9c9mp6w7m/ogbg_molhiv.zip"
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