

# Command to download dataset:
#   bash script_download_all_datasets.sh



############
# ZINC
############

DIR=molecules/
cd $DIR

FILE=ZINC.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/bhimk9p1xst6dvo/ZINC.pkl?dl=1 -o ZINC.pkl -J -L -k
fi

cd ..


############
# MNIST and CIFAR10
############

DIR=superpixels/
cd $DIR

FILE=MNIST.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/wcfmo4yvnylceaz/MNIST.pkl?dl=1 -o MNIST.pkl -J -L -k
fi

FILE=CIFAR10.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/agocm8pxg5u8yb5/CIFAR10.pkl?dl=1 -o CIFAR10.pkl -J -L -k
fi

cd ..


############
# PATTERN and CLUSTER 
############

DIR=SBMs/
cd $DIR

FILE=SBM_CLUSTER.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/edpjywwexztxann/SBM_CLUSTER.pkl?dl=1 -o SBM_CLUSTER.pkl -J -L -k
fi

FILE=SBM_PATTERN.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/9h6crgk4argc89o/SBM_PATTERN.pkl?dl=1 -o SBM_PATTERN.pkl -J -L -k
fi

cd ..


############
# SGS
############

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

cd ..
