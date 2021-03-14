## pull ARFF data (restaurant data by default) into a pandas dataframe
## Note - this has only been tested with restaurant.arff.


def getARFFData(filename='restaurant.arff') :
    with open(filename) as rest :
        lines = rest.readlines()

        attrs = [line.strip('\n') for line in lines if line.startswith('@attribute')]
        attrNames = [line.strip('\n').split()[1] for line in lines if line.startswith('@attribute')]
        dataIndex = lines.index('@data\n')
        data = [line.strip('\n').split(', ') for line in lines[dataIndex+1:]]

        ### now let's make the pandas dataframe
        df = pd.DataFrame(data, columns=attrNames)

        ## let's make a dictionary that maps atrtribute names to possible values.
        attributeDict = {}
        for a in attrs :
            name = a.split()[1]
            vals = a[a.find('{')+1:a.find('}')].split(', ')
            attributeDict[name] = vals
    return df, attributeDict




