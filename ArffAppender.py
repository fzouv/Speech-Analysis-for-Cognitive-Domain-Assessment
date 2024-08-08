# # # Fasih's code
# import os
# import glob
# import builtins
# import re
# from natsort import natsorted, ns
# import configparser
#
# config = configparser.ConfigParser()
# config.read('paths.ini')
# arffFolder = config['DEFAULT']['OUTPUT_FOLDER']
#
#
# def main():
#     comb_arffFileName = "combinedARFF.csv"  # please enter file name
#
#     arff_files = [
#         filename
#         for filename
#         in os.listdir(arffFolder)
#         if filename.endswith('.arff')
#     ]
#
#     arff_files = natsorted(arff_files, alg=ns.IGNORECASE)
#     combined_arffFile = open(comb_arffFileName, "w")
#     headers_read = False
#     for file in arff_files:
#         print(file)
#         arff_file = open(arffFolder + file, "r")
#         if headers_read == False:
#             for line in arff_file:
#                 # combined_arffFile.write(line)
#                 if line.find("liveturn_0") == 1:  # "unknown" for egemaps
#                     print(file + ',' + line[10:-3])
#                     combined_arffFile.write(file + ',' + line[10:-3] + '\n')
#                 if line.find("unknown") == 1:  # "unknown" for egemaps
#                     print(file + ',' + line[10:-3])
#                     combined_arffFile.write(file + ',' + line[10:-2] + '\n')
#             #  print("header's file written\n")
#
#             headers_read = True
#         else:
#             reached_data = False
#             for line in arff_file:
#                 if line.strip() == "@data":
#                     reached_data = True
#                     continue
#                 if reached_data == True:
#                     if line == "\n":
#                         continue
#                     combined_arffFile.write(file + ',' + line[10:-3] + '\n')
#                     # print(file)
#     #     print ("file appended\n")
#     # print ("Combined file written\n")
#
#
# if __name__ == '__main__':
#     status = main()
#     # sys.exit(status)

#code version 2 -- for headers (but did not handle 'unknown')
import os
import glob
import builtins
import re
import configparser
from natsort import natsorted, ns

def main():
    # Load configuration and paths
    config = configparser.ConfigParser()
    config.read('paths.ini')
    arffFolder = config['DEFAULT']['OUTPUT_FOLDER']

    # Define the combined output file name
    comb_arffFileName = os.path.join(arffFolder, "combinedARFF.csv")

    # Get all ARFF files in the specified directory
    arff_files = [os.path.join(arffFolder, f) for f in os.listdir(arffFolder) if f.endswith('.arff')]
    arff_files = natsorted(arff_files, alg=ns.IGNORECASE)

    # Prepare to write combined CSV
    with open(comb_arffFileName, "w") as combined_arffFile:
        headers_written = False

        for file in arff_files:
            with open(file, "r") as arff_file:
                headers_read = False
                for line in arff_file:
                    if line.strip().startswith('@data'):
                        headers_read = True
                        continue  # Skip the '@data' line and stop processing headers
                    if not headers_read:
                        if not headers_written:  # Write headers only once
                            if line.startswith('@attribute'):
                                header = line.split(' ')[1]
                                combined_arffFile.write(header + ',')
                            continue
                    else:
                        # Write data
                        combined_arffFile.write('\n' + file + ',' + line.strip())

                headers_written = True  # Set to True after first file's headers are written

if __name__ == '__main__':
    main()



