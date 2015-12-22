import os

fileCount = 0
maxFiles = 2500 
for filename in os.listdir('.'):
    if not os.path.isfile(filename):
        continue
    if not filename.endswith('.txt'):
        continue

    print(fileCount)
    if fileCount > maxFiles:
        break

    fileHeader = True
    with open(filename) as file:
        count = 0
        fileCount = fileCount + 1

        if fileCount > maxFiles:
            break

        for line in file:
            if not fileHeader:
                count = count + 1
            else:
                fileHeader = False

        #Go back to top of file
        file.seek(0)
        training = open('training/' + filename, 'w+')
        validation = open('validation/' + filename, 'w+')

        fileHeader = True
        ratingCount = 0
        for line in file:
            if not fileHeader:
                ratingCount = ratingCount + 1
                if ratingCount <= 0.8 * count:
                    training.write(line)
                else:
                    validation.write(line)
            else:
                training.write(line)
                validation.write(line)
                fileHeader = False
        training.close()
        validation.close()
        file.close()



