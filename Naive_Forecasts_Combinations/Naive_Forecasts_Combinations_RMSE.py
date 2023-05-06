#   ---------FIFTH CLASS EXERCISES -----------
#               Report task 2
#             till 17th of April
#           ---------------------
#              SEBASTIAN SUWADA 
#           ---------------------
#

import numpy as np 
import matplotlib.pyplot as plt
import random
import math
import pandas as pd
from datetime import date, timedelta
import itertools


#---------------- FUNC ---------------

def load_df():
    df = pd.read_csv(r'Python_Lab_BI/vic_elec_forecast_corr.csv', parse_dates=['Time'])
    index_col = 0
    return df

def naive1(df):
    df['Naive_forecast1'] = df['Demand'].shift(1)
    return df

def naive2(df):
    df['Naive_forecast2'] = df['Demand'].shift(48)
    return df

def naive3(df):

    demand1 = df['Demand'].to_numpy()
    holiday = df['Holiday'].to_numpy()
    dates = df['Date'].to_numpy()

    #   Shift 48 postions 
    df['Naive_forecast3'] = df['Demand'].shift(48)

    search_not_holi = []
    search_holiday = []
    search_sunday = []

    #   Create list of positions of holiday
    #   Create list of positions of Sunday
    #   Create list of positions of not holiday
    #   Create index of week days, and transforme it into numpy

    df['WeekdayIndex'] = df['Time'].dt.dayofweek
    navi3 = df['Naive_forecast3'].to_numpy()
    weekday = df['WeekdayIndex'].to_numpy()

    for i in range(0,len(demand1),48):
        if holiday[i] == 1:
            search_holiday.append(i)
        else:
            search_not_holi.append(i)
        if weekday[i] == 6:
            search_sunday.append(i)


    searchh = []
    searchd = []
    countk = 0

    #   Create check what was first - Sunday or Holiday and assigne coressponding values
    #   into list Navi3 -> which is a numpy value of dataframe from Naive forecast 3. 

    for j in range(0,len(demand1),48):    
        
        searchd.clear()
        searchh.clear()

        if holiday[j] == 1:

            for k in range(len(search_holiday)):
                if search_holiday[k]<j:
                    searchh.append(search_holiday[k])
                else:
                    break
            for b in range(len(search_sunday)):
                if search_sunday[b]<j:
                    searchd.append(search_sunday[b])
                else:
                    break

            if len(searchd) != 0 and len(searchh) != 0:
                if max(searchh)>max(searchd):
                    for i in range(max(searchh),max(searchh)+48,1):
                        temp = max(searchh)-i
                        navi3[j-temp] = demand1[i]
                if max(searchd)>max(searchh):
                    for i in range(max(searchd),max(searchd)+48,1):
                        temp = max(searchd) - i
                        navi3[j-temp] = demand1[i]            
            else:
                for i in range(countk,j,1):
                    navi3[i] = np.nan
                #countk = countk + 48

    #   48 can be search and assigned by searching for value that was sampled from data
    #   48 is sample of data frame 

    return df

def compare_dates(demand,df):
    dates = df['Date'].to_numpy(dtype='datetime64')
    compare_date = np.array(['2012-12-31'], dtype='datetime64')
    naive1 = df['Naive_forecast1'].to_numpy()
    naive2 = df['Naive_forecast2'].to_numpy()
    naive3 = df['Naive_forecast3'].to_numpy()

    o1314_naive1 = []
    o1314_naive2 = []
    o1314_naive3 = []
    demand2 = []

    for i in range(len(demand)):
        if dates[i] > compare_date[0]:
            o1314_naive1.append(naive1[i])
            o1314_naive2.append(naive2[i])
            o1314_naive3.append(naive3[i])
            demand2.append(demand[i])
    return o1314_naive1,o1314_naive2, o1314_naive3, demand2

def compute_mae_rmse(df):
     
    demand = df['Demand'].to_numpy()


    o1314_naive1 = []
    o1314_naive2 = []
    o1314_naive3 = []

    o1314_naive1, o1314_naive2, o1314_naive3, demand2 = compare_dates(demand,df)

    mae1 = np.mean(np.abs(np.array(o1314_naive1) - np.array(demand2)))
    rmse1 = np.sqrt(np.mean((np.array(o1314_naive1) - np.array(demand2))**2))

    mae2 = np.mean(np.abs(np.array(o1314_naive2) - np.array(demand2)))
    rmse2 = np.sqrt(np.mean((np.array(o1314_naive2) - np.array(demand2))**2))

    mae3 = np.mean(np.abs(np.array(o1314_naive3) - np.array(demand2)))
    rmse3 = np.sqrt(np.mean((np.array(o1314_naive3) - np.array(demand2))**2))


    print('--------------------------- MAE AND RMSE Results -----------------------'+'\n')

    print('MAE Naiveforecast 1 ->'+str(mae1)+'\n')
    print('MAE Naiveforecast 2 ->'+str(mae2)+'\n')
    print('MAE Naiveforecast 3 ->'+str(mae3)+'\n')
    print('RMSE Naiveforecast 1 ->'+str(rmse1)+'\n')
    print('RMSE Naiveforecast 2 ->'+str(rmse2)+'\n')
    print('RMSE Naiveforecast 3 ->'+str(rmse3)+'\n')

    print('------------------------------  End  -----------------------'+'\n')

def Average(list1):
     su = sum(list1)
     l = len(list1)
     avg = su/l
     return avg

def create_average_list(demand,naive):
    
    week = df['WeekdayIndex'].to_numpy()    
    holiday = df['Holiday'].to_numpy()

    search_holiday_pos = []
    search_not_holi_pos = []

#   Get list of positions where is holiday or not
    #   This line is giving us place(position) for 01.01.2013 -> [len(demand)-len(o1314_naive1)] = pos1
    #   Then we have range of i1 = pos1, range(lenght of demand)
    #   search_holiday_pos -> list of positions of holidays
    #   search_not_holi_pos -> list of positions of not holidays

    for i in range(len(demand)-len(naive),len(demand),1):
        if holiday[i] == 1:
            search_holiday_pos.append(i)
        if holiday[i] == 0:
            search_not_holi_pos.append(i)

    #   Compute average of not holidays
    #   There we play about positions of holiday or not holiday, and create two lists
    #   where one is temporary and is cleared when we change day - cleared for another day

    list_for_avg_n = []
    list_for_avg = []

    demand_list_for_avg_n = []
    demand_list_for_avg = []

    avg_list_days_demand = []
    avg_list_days_demand_n = []

    avg_list_days_n = []
    avg_list_days_holi = []

    temp = 0
    temp2 = 0
    temp3 = 0
    temp4 = 0
    
    #   Holidays average!

    for j in range(len(search_holiday_pos)-1):
        list_for_avg.append(naive[search_holiday_pos[j]])
        demand_list_for_avg.append(demand[search_holiday_pos[j]])
        if week[search_holiday_pos[j]] != week[search_holiday_pos[j+1]]: #or j == len(search_not_holi_pos)-1:
            temp = Average(list_for_avg)
            temp2 = Average(demand_list_for_avg)
            #print(temp)
            
            avg_list_days_demand.append(temp2)
            avg_list_days_holi.append(temp)
            list_for_avg.clear()
            demand_list_for_avg.clear()

    #    NOT Holidays average!

    for j in range(len(search_not_holi_pos)-1):
        list_for_avg_n.append(naive[search_not_holi_pos[j]])
        demand_list_for_avg_n.append(demand[search_not_holi_pos[j]])
        #print(list_for_avg[:10])
        #print(week[search_not_holi_pos[j]], week[search_not_holi_pos[j+1]])
        if week[search_not_holi_pos[j]] != week[search_not_holi_pos[j+1]]: #or j == len(search_not_holi_pos)-1:
            temp3 = Average(list_for_avg_n)
            temp4 = Average(demand_list_for_avg_n)
            #print(temp)
            
            avg_list_days_demand_n.append(temp4)
            avg_list_days_n.append(temp3)
            list_for_avg_n.clear()
            demand_list_for_avg_n.clear()

    
    return avg_list_days_n, avg_list_days_holi, avg_list_days_demand_n, avg_list_days_demand

def average_of_days(df):

    demand = df['Demand'].to_numpy()
    #naive1 = df['Naive_forecast1'].to_numpy()
    #dates = df['Date'].to_numpy(dtype='datetime64')

    naive1_avg_n = []
    naive1_avg_holi = []

    naive2_avg_n = []
    naive2_avg_holi = []

    naive3_avg_n = []
    naive3_avg_holi = []

    demand_avg_list_n = []
    demand_avg_list_holi = []


    #   Create 3 list where we have values of naive1-3 and 
    #   demand that is only for years 2013 and 2014

    o1314_naive1, o1314_naive2, o1314_naive3, demand2 = compare_dates(demand,df)

    #   For each naive we create average list of calculated average value for 
    #   each day if it is holiday or its not
    #   naiveX_avg_n -> list of naive(X) averages of one X-day [FOR NOT HOLIDAY]
    #   naiveX_avg_holi -> list of naive(X) averages of one X-day [FOR HOLIDAYS]
    #   avg_list_demand_days_holi/n -> list of average from each day depending if its holiday or not and
    #                                   if it comes from naive1 or naive 3, also first output is not holiyday second is
    #                                   for holiday
    #  return(naive_avg_notholiday, naive_avg_holiday, demand_list_avg_notholiday, demand_list_avg_holiday)

    naive1_avg_n, naive1_avg_holi, demand_avg_list_n, demand_avg_list_holi = create_average_list(demand2,o1314_naive1)
    naive2_avg_n, naive2_avg_holi, demand_avg_list_n, demand_avg_list_holi = create_average_list(demand2,o1314_naive2)
    naive3_avg_n, naive3_avg_holi, demand_avg_list_n, demand_avg_list_holi = create_average_list(demand2,o1314_naive3)

    # print(len(naive1_avg_holi), len(naive1_avg_n))
    # print(len(naive2_avg_holi), len(naive2_avg_n))
    # print(len(naive3_avg_holi), len(naive3_avg_n))

    # calculate MAPE
    mape_naive1_n = []
    mape_naive2_n = []
    mape_naive3_n = []

    mape_naive1_h = []
    mape_naive2_h = []
    mape_naive3_h = []


    for i in range(len(demand_avg_list_n)):
        mape_naive1_n.append(np.mean(np.abs((demand_avg_list_n[i] - naive1_avg_n[i]) / demand_avg_list_n[i])) * 100)
        mape_naive2_n.append(np.mean(np.abs((demand_avg_list_n[i] - naive2_avg_n[i]) / demand_avg_list_n[i])) * 100)
        mape_naive3_n.append(np.mean(np.abs((demand_avg_list_n[i] - naive3_avg_n[i]) / demand_avg_list_n[i])) * 100)
    #print(mape_naive2_n[1])
    #print(mape_naive3_n[1])


    for i in range(len(demand_avg_list_holi)):
        mape_naive1_h.append(np.mean(np.abs((demand_avg_list_holi[i] - naive1_avg_holi[i]) / demand_avg_list_holi[i])) * 100)
        mape_naive2_h.append(np.mean(np.abs((demand_avg_list_holi[i] - naive2_avg_holi[i]) / demand_avg_list_holi[i])) * 100)
        mape_naive3_h.append(np.mean(np.abs((demand_avg_list_holi[i] - naive3_avg_holi[i]) / demand_avg_list_holi[i])) * 100)
    #print(mape_naive2_h[1])
    #print(mape_naive3_h[1])

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].boxplot([mape_naive1_h, mape_naive1_n], labels=['Holiday', 'Non-holiday'])
    axs[0].set_title('Naive Forecast 1')
    axs[1].boxplot([mape_naive2_h, mape_naive2_n], labels=['Holiday', 'Non-holiday'])
    axs[1].set_title('Naive Forecast 2')
    axs[2].boxplot([mape_naive3_h, mape_naive3_n], labels=['Holiday', 'Non-holiday'])
    axs[2].set_title('Naive Forecast 3')

    plt.show()

def compare_date_2(demand):
    #   Get positions of 2013
    dates = df['Date'].to_numpy(dtype='datetime64')
    compare_date_min = np.array(['2013-01-01'], dtype='datetime64')
    compare_date_max = np.array(['2013-12-31'], dtype='datetime64')
    positions_2013 = []
    for i in range(len(demand)):
        if dates[i] >= compare_date_min[0] and dates[i] <= compare_date_max[0]:
            positions_2013.append(i)
    
    #print(dates[positions_2013[0]])
    return positions_2013

def combintaions_calc(forecasts_list,demand):

    combinations_all = []
    combinations_all_val = []
    avg_forecast = []
    rmse_scores = []
    comb = []

    # a = [1,2,3,4]
    # b = [5,6,7,8]
    # c = [9,10,11,12]
    # d = [13,14,15,16]

    # forec = [a,b,c,d]
    combination_values = [1,2,3,4,5,6]

    for r in range(1, len(combination_values)+1):
        for combination in itertools.combinations(combination_values, r):
            if len(combination) == r:
                combinations_all_val.append(combination)
    #print(combinations_all_val)

    for r in range(1,len(forecasts_list)+1):
        for combination in itertools.combinations(forecasts_list, r):
            if len(combination) == r:
                combinations_all.append(combination)

    #   Convert tuple -> list

    for h in range(len(combinations_all)):
        merged = list(itertools.chain(*combinations_all[h]))    #   Merge list in list of list into list of lists 
        temp = list(merged)                                     #   Convert into list from tuple using list()
        comb.append(temp)

    len_comb = len(comb)

    # rmse_indexes = []
    # for c in comb:
    #     avg_combined = sum(c)/len(c)
    #     rmsev2 = np.sqrt(np.mean(np.square(np.array(avg_combined) - np.array(demand))))
    #     rmse_indexes.append(rmsev2)

    # print(rmse_indexes)

    #   Calculate average of values from lists, and for demand in loop
    #   Calculate RMSE value and create list of those values.

    arr = np.vstack(forecasts_list).T
    #print(arr.shape)
    #print(combination_values)

    for i in range(len(comb)): #range(len(comb)):
        avg_forecast = arr[:,[e-1 for e in combinations_all_val[i]]].mean(axis = 1)
        rmse = np.sqrt(np.mean(np.square((avg_forecast - np.array(demand)))))
        rmse_scores.append(rmse)

    print('All RMSE Scores: '+'\n')
    print(rmse_scores)


    #   PLOT histogram

    plt.hist(rmse_scores)

    # Add axis labels and title

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of RMSE with '+str(len_comb)+' combinations \n')    
    plt.show()

    best_rmse_index = rmse_scores.index(min(rmse_scores))
    #best_rmse = rmse_scores[best_rmse_index]
    print("\n"+"Forecasts on list are: 1 - Naive 1 ; 2 - Naive 2 ; 3 - Naive 3"+"\n"+"4 - Demand Forecast 1 ; 5 - Demand Forecast 2 ; 6 - Demand Forecast 3"+"\n")
    print('Best RMSE index is ->'+str(best_rmse_index)+'\n')
    print('Forecasts taken ->'+str(combinations_all_val[best_rmse_index])+'\n')    

    return combinations_all_val[best_rmse_index]

def compute_rmse(df):

    #   Define values from pandas to numpy

    naive1 = df['Naive_forecast1'].to_numpy()
    naive2 = df['Naive_forecast2'].to_numpy()
    naive3 = df['Naive_forecast3'].to_numpy()
    demandf1 = df['Demand_forecast1'].to_numpy()
    demandf2 = df['Demand_forecast2'].to_numpy()
    demandf3 = df['Demand_forecast3'].to_numpy()
    demand = df['Demand'].to_numpy()

    positions = []

    #   Get positions of dates for 2013

    positions = compare_date_2(demand)

    #   Compute RMSE value
    #   Rmse = np.sqrt(np.mean(np.square(y_pred - y_actual)))

    naive1_2013 = []
    naive2_2013 = []
    naive3_2013 = []

    demandf1_2013 = []
    demandf2_2013 = []
    demandf3_2013 = []

    demand2 = []


    for i in range(positions[0],positions[-1],1):
        naive1_2013.append(naive1[i])
        naive2_2013.append(naive2[i])
        naive3_2013.append(naive3[i])
        demandf1_2013.append(demandf1[i])
        demandf2_2013.append(demandf2[i])
        demandf3_2013.append(demandf3[i])
        demand2.append(demand[i])


    rmse_naive1 = np.sqrt(np.mean(np.square(np.array(naive1_2013) - np.array(demand2))))
    rmse_naive2 = np.sqrt(np.mean(np.square(np.array(naive2_2013) - np.array(demand2))))
    rmse_naive3 = np.sqrt(np.mean(np.square(np.array(naive3_2013) - np.array(demand2))))
    rmse_df1 = np.sqrt(np.mean(np.square(np.array(demandf1_2013) - np.array(demand2))))
    rmse_df2 = np.sqrt(np.mean(np.square(np.array(demandf2_2013) - np.array(demand2))))
    rmse_df3 = np.sqrt(np.mean(np.square(np.array(demandf3_2013) - np.array(demand2))))

    print('--------------------- RMSE of the half-hourly values for all  -----------------------'+'\n')

    print('RMSE Naiveforecast 1 ->'+str(rmse_naive1)+'\n')
    print('RMSE Naiveforecast 2 ->'+str(rmse_naive2)+'\n')
    print('RMSE Naiveforecast 3 ->'+str(rmse_naive3)+'\n')
    print('RMSE Demand forecast 1 ->'+str(rmse_df1)+'\n')
    print('RMSE Demand forecast 2 ->'+str(rmse_df2)+'\n')
    print('RMSE Demand forecast 3 ->'+str(rmse_df3)+'\n')

    print('------------------------------ End  -----------------------'+'\n')

    # Create a list of all possible combinations of 1 to 6 forecasts

    #forecasts = [naive1,naive2,naive3,demandf1,demandf2,demandf3]
    forecasts = [naive1_2013,naive2_2013,naive3_2013,demandf1_2013,demandf2_2013,demandf3_2013]

    index = 0

    index = combintaions_calc(forecasts,demand2)

    #   The MAE of the best combination in {MONTH} 2014 is {RESULT}.

    dates = df['Date'].to_numpy(dtype='datetime64')
    compare_date_min = np.array(['2014-08-01'], dtype='datetime64')
    compare_date_max = np.array(['2014-08-31'], dtype='datetime64')
    positions_agust = []
    for i in range(len(demand)):
        if dates[i] >= compare_date_min[0] and dates[i] <= compare_date_max[0]:
            positions_agust.append(i)

    naive1_2014A = []
    naive2_2014A = []
    naive3_2014A = []

    demandf1_2014A = []
    demandf2_2014A = []
    demandf3_2014A = []

    demand3 = []

    for i in range(positions_agust[0],positions_agust[-1],1):
        naive1_2014A.append(naive1[i])
        naive2_2014A.append(naive2[i])
        naive3_2014A.append(naive3[i])
        demandf1_2014A.append(demandf1[i])
        demandf2_2014A.append(demandf2[i])
        demandf3_2014A.append(demandf3[i])
        demand3.append(demand[i])
    
    #   BIRTH MONTH EXERCISE !!!

    forecast_2 = [naive1_2014A,naive2_2014A, naive3_2014A,demandf1_2014A,demandf2_2014A,demandf3_2014A]
    index = list(index)
    #print(index)

    extended_list = []

    for i in range(len(index)):
        extended_list.extend(forecast_2[index[i]-1])

    for l in range(len(index)-2):
        demand3.extend(demand3)
    mae_bith = np.mean(np.abs(np.array(extended_list) - np.array(demand3)))

    print('The MAE of the best combination in AGUST 2014 is -> '+str(mae_bith)+'\n')

#---------------- MAIN ---------------

#       Import data from csv
#       REPORT TASK EX.
#       1. Download the dataset with 3 point forecasts from List 5 
#       (named L02b (List 5) point forecasts in ePortal, under T02+L02 Time series graphics). 
#       Load (using either numpy or pandas) the contents of the file and:

df = load_df()

#       a) prepare a new column with a naive forecast that takes the value of the last half-hourly 
#       real observation as a forecast for the next one. 
#       Note that you don’t have the value for the first half-hourly observation
#       Bierzemy wartosc poprzedzajaca o okres czyli o 30 min

naive1(df)

#       b) przepisujemy wartosc poprzedzajaca o dzien do tyl zgodnie dla kazdego czasu.
#       df.iloc[1:,-1] = df.iloc[:-1,0] -> wektorowe zrobienie dzialania dla naiva
#       w c robimy to samo tylko uwzgledniamy godziny swiateczne

naive2(df)

#        c) prepare a new column with a third naive forecast: let it be equal to:
#           1. a value from the corresponding half-hourly observation of the preceding day for days that are not holidays
#           2. a value from the corresponding half-hourly observation of the last Sunday or the last holiday, whichever is closer to the forecasted date

naive3(df)

#       Compute the error metrics (MAE and RMSE) of all three 
#       naive forecasts for all data points from years 2013 and 2014

compute_mae_rmse(df)

#       For each of the naive forecasts, prepare a boxplot of the of the MAPE error metric of the daily 
#       averages (note that it differs from the daily average of the MAPE errors) for all data points 
#       from years 2013 and 2014 depending on whether the day is a holiday (the boxplot should have two 
#       “boxes” – one describing the holidays and one describing the rest of the days). 
#       Place the plots in three separate subplots in one figure 
#       (the figure should contain one row of three columns)

average_of_days(df)

#       Compute the RMSE scores of the half-hourly values for all 3 naive methods and all 
#       3 forecasts from the file for year 2013. Print them out in your program, then find 
#       the best equally-weighted combination of the 6 forecasts (for every possible combination 
#       of 1 to 6 forecasts, compute the combined forecast – an equally weighted average of them – 
#       and compute its RMSE). Plot a histogram of the RMSE values for all combined forecast,
#       in its title place information on the number of all combinations.

compute_rmse(df)




#  -------------------------------------- BRUDNOPIS -----------------------------------

# start_date = date(2012, 1, 1)
# end_date = date(2014, 12, 31)
# delta = timedelta(days=1)
# while start_date <= end_date:
#     #print(start_date.weekday())
#     start_date += delta


#print(df.dtypes)
#Print whole data set
# print(df.to_string())

#       pandas to numpy -> df.to_numpy() , df.loc[]... , dt.weekday() - > zwroci numerek od zera do 6

    #Drop all rows that have NaN value
    #df2=df.dropna()
    #df2=df.dropna(axis=0)
    # Reset index after drop
    #df2=df.dropna().reset_index(drop=True)

    #plt.figure(1)
    #plt.boxplot(naive2)
    #plt.show()

    #plt.figure(2)
    #plt.boxplot(naive1)
    #plt.show()

#   NAIVE 3!
#   idziemy do tylu i sprawdzamy co bylo pierwszze niedziela czy swieto jakies jezeli swieto
#   to nadpisujemy holiday ktory mamy tym co bylo pierwsze w tyl. Dla wartosci 1 stycznia 2012 beda nany

#   zadanie (D) - error dla 2013 i 2014 tylko z tych naivow!!!

#   Zadanie (E) - liczymy srednia dla przebiegu dobowego czyli 48 wartosci polgodzinnych i napodstawie tego 
#   obliczamy ten blad i robimy boxplota z tym bledem czyli beda 3 figury bo 3 naivy, ale na jednej beda dwa slupki,
#   jeden dla holiday a drugi dla nie holiday!, Nalezy pamietac ze to ma byc tylko dla lat 2013 i 2014!!

#   Zadanie (F i G) - mamy 6 prognoz i dla kazdej z prognoz liczymy kombinacje czyli bedzie kombinacja np.
#   1 z druga, 2 z trecia itp, moga byc tez wszystkie, dla kazdej z kombinacji liczymy blad RMSE, i sprawdzamy 
#   ktory jest najnizszy dla najnizszego w kolejnym punkcie dla roku 2014 liczymy dla naszego miesiaca to samo tylko
#   dla innego bledu. W poprzednim trzeba jeszcze zrobic histogram dla wartosci bledow.