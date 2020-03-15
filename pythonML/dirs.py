import os
import shutil
data_dir = "C:/git/pythonML/pythonML/notebooks/kaggle/plankton/"
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')

dir_info = [x for x in os.walk(train_dir)]
for dir in dir_info[1:]:
    print(len(dir[2]))
    validation_number = int(round(len(dir[2])/5))
    print(validation_number)
    print(dir[2])
    print(dir[2][:validation_number])
    dir_name = os.path.split(dir[0])[1]
    print(dir_name)
    validation_dir_current = os.path.join(validation_dir, dir_name)
    if os.path.exists(validation_dir_current) == False:
        os.mkdir(validation_dir_current)
    for fname in dir[2][:validation_number]:
        src = os.path.join(train_dir, dir_name, fname)
        dst = os.path.join(validation_dir_current, fname)
        print(src)
        print(dst)
        shutil.move(src, dst)
