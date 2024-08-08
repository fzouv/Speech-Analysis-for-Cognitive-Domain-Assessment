import os
import os.path
import sys
import subprocess
import configparser
config = configparser.ConfigParser()
config.read('paths.ini')
INPUT_FOLDER=config['DEFAULT']['INPUT_FOLDER']
OUTPUT_FOLDER=config['DEFAULT']['OUTPUT_FOLDER']
OPEM_SMILE=config['DEFAULT']['OPEM_SMILE']
FEATURE=config['DEFAULT']['FEATURE']
def main():
    path = os.getcwd()
    filenames = [
        filename
        for filename
        in os.listdir(INPUT_FOLDER)
        if filename.endswith('.wav')
        ]

    for filename in filenames:
        outfile=filename[:-4]+'.arff';
        subprocess.call([
            OPEM_SMILE, "-C", FEATURE, "-I", os.path.join(INPUT_FOLDER, filename), "-O", os.path.join(OUTPUT_FOLDER, outfile)
            ])

    os.system('python ArffAppender.py')
    return 0



if __name__ == '__main__':
   status = main()
    #sys.exit(status)
