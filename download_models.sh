# Download the answering model
mkdir models/answering
cd models/answering
gdown https://drive.google.com/uc?id=1q2Z3FPP9AYNz0RJKHMlaweNhmLQoyPA8
unzip model.zip
rm model.zip
cd ../..

# Download the classifier 
mkdir models/classification
cd models/classification
gdown https://drive.google.com/file/d/1PuQA6bYsnKrIUU1i3ioKhr7xxWnJhdSh/view?usp=drive_link
unzip learned_classifier.zip
rm learned_classifier.zip
cd ../..
