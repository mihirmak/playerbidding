
import pandas as pd
import numpy as np

data=pd.read_csv("mihir.csv")//Contains predicted values of the players price

values = np.asarray(data[['MLPRegressor']])
players = np.asarray(data[['Player']])



nplayers = players.size


nteams = 4
counter = 0
#values = [50, 60, 70, 80, 90, 100]
#players = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
base_price = []
bid = [0]*nteams
out = [0]*nteams
fteams = [[], [], [], []]
pocket = [500000, 500000, 500000, 500000]
players = [0]* nteams
for i in range(nplayers):
    if values[i] <= 60:
        base_price.append(10000)
    elif values[i] >= 61 and values[i] <= 73:
        base_price.append(20000)
    elif values[i] >= 74 and values[i] <= 80:
        base_price.append(30000)
    elif values[i] >= 81 and values[i] <= 90:
        base_price.append(40000)
    else:
        base_price.append(50000)
j = 0
z = 0
for i in range(nplayers):
    counter = 0
    print("\n")
    print("For player",i+1)
    j = 0
    c = 0
    z = 0
    while out.count(1) < (nteams):
        j = j % nteams
        print("\n")
        print("BID FOR TEAM :",j+1)
        print("THE CURRENT BID IS :",max(bid)+base_price[i])
        print("AVAILABLE MONEY IN YOUR POCKET :",pocket[j])
        if len(fteams[j]) == 11:
            print("YOU CANNOT BID AS YOU ALREADY HAVE 11 PLAYERS")
            out[j] = 1
            c += 1
            j += 1

        else:
            select = input("Press 'Y' if you want to bid and 'N' if you dont!")
            c += 1
            if select=='Y' or select=='y':
                choice = int(input("Enter the prize you wanna bid"))
                if (choice + base_price[i])>pocket[j]:
                    print("CANNOT BID, DO NOT HAVE ENOUGH MONEY")
                    out[j] = 1
                    if out.count(1) == nteams-1:
                        pos = out.index(0)
                        print("Team" +str(pos+1)+" gets player"+str(i+1)+" for "+str(max(bid)))
                else:
                    if choice <= max(bid)+base_price[i]:
                        print("THE BID CANNOT BE LESS THAN",max(bid))
                        j -= 1
                    else:
                        bid[j] = choice
            else:
                print("I am out for this player")
                out[j] = 1
                if (out.count(1)==(nteams - 1) and c >= (nteams)) or out.count(1)==(nteams):
                    if (0 in out):
                        pos = out.index(0)
                        print("Team" +str(pos+1)+" gets player"+str(i+1)+" for "+str(max(bid)))
                        fteams[pos].append(i+1)
                        pocket[pos] -= (bid[pos]+base_price[i])
                        bid = [0]*4
                        out = [0]*4
                        print(pocket)
                        break
                    else:
                        print("THE PLAYER IS UNSOLD!")
                        out = [0] * 4
                        print(pocket)
                        break
            j = j + 1
    print(fteams)
    for z in range(nteams):
        print(counter)
        if len(fteams[z]) == 11:
            counter += 1

    if counter == nteams:
        print("The teams have been formed!!")
        break