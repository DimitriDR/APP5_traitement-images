def accuracy_checking(experimental_file: str, truth_file: str):
    with open(truth_file, "r") as truth:
        truth_file_content = [line.split() for line in truth]

    with open(experimental_file, "r") as experimental:
        experimental_file_content = [line.split() for line in experimental]

    truth_file_content = {int(line[0]): line[1:] for line in truth_file_content}
    experimental_file_content = {int(line[0]): line[1:] for line in experimental_file_content}

    print("truth" + str(len(truth_file_content)))
    print("exp" + str(len(experimental_file_content)))
    print(truth_file_content)

    nb_fragments_placed = len(experimental_file_content)
    nb_fragments = len(truth_file_content)

    print("nb_fragments_placed: " + str(nb_fragments_placed))
    print("nb_fragments: " + str(nb_fragments))

    nb_correct_fragments = 0

    pixel_error = 50
    angle_error = 5
    for index, coordinates in truth_file_content.items():
        if index in experimental_file_content:
            if (abs(float(experimental_file_content[index][0]) - float(coordinates[0])) <= pixel_error
                    and abs(float(experimental_file_content[index][1]) - float(coordinates[1])) <= pixel_error
                    and abs(float(experimental_file_content[index][2]) - float(coordinates[2])) <= angle_error):
                print("Fragment " + str(index) + " is correct")
                print("Experimental: " + str(experimental_file_content[index]))
                print("Truth: " + str(coordinates))
                nb_correct_fragments += 1
            else:
                print("Fragment " + str(index) + " is incorrect")
                print("Experimental: " + str(experimental_file_content[index]))
                print("Truth: " + str(coordinates))

    print("nb_correct_fragments: " + str(nb_correct_fragments))
    accuracy = round(nb_correct_fragments / nb_fragments * 100, 2)
    print("Precision : " + str(accuracy) + "%")


accuracy_checking(f"./reconstruction_results.txt", f"./solution_fragments.txt")
