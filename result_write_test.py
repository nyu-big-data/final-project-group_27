
import code.constants as const

file_output_string = "This is a test"
with open(const.RESULTS_SAVE_FILE_PATH, 'a') as output_file:
            print("Recording the following: model_params")
            output_file.write(f"{file_output_string}")