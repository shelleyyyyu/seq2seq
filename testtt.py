with open("./result/result.txt", "r") as f:
    line = f.readline()
    arr = line.split("#")
    print(len(arr))
    print(arr[0])

