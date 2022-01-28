import os

result_folder = "./SimulationResults"
save_address = "./SimulationResults"
MovieLens_save_address = './MovieLensResults'
datasets_address = '.'  # should be modified according to the address of data
MovieLens_address = datasets_address + '/Dataset/ml-20m/processed_data'
MovieLens_FeatureVectorsFileName = os.path.join(MovieLens_address, 'Arm_FeatureVectors_2.dat')
MovieLens_relationFileName = os.path.join(MovieLens_address, 'user_contacts.dat.mapped')