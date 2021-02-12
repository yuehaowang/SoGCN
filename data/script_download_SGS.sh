

# Command to download dataset:
#   bash script_download_SGS.sh


DIR=SGS/
cd $DIR


FILE=SGS_BAND_PASS.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/y2ynt6ufuueof1u/SGS_BAND_PASS.pkl?dl=1 -o SGS_BAND_PASS.pkl -J -L -k
fi


FILE=SGS_HIGH_PASS.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/w77cdmf0ththpk8/SGS_HIGH_PASS.pkl?dl=1 -o SGS_HIGH_PASS.pkl -J -L -k
fi


FILE=SGS_LOW_PASS.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/jfcnadproazh816/SGS_LOW_PASS.pkl?dl=1 -o SGS_LOW_PASS.pkl -J -L -k
fi

