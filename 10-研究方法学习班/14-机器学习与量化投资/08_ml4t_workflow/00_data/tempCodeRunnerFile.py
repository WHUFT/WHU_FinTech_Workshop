prices = (store['quandl/wiki/prices']
        #           .filter(like='adj')
        #           .rename(columns=lambda x: x.replace('adj_', ''))
        #           .swaplevel(axis=0))
        # print(prices.head())