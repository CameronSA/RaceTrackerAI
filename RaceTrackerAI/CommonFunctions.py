def AddToDict(dict,key,value):
    if key in dict:
        dict[key].append(value)
    else:
        dict[key] = [value]
