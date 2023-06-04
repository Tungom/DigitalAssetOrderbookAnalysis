
#--------Import Libraries --------------
import requests 
import json 
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import time
import numpy as np
from pybit.unified_trading import HTTP
#----------------------------------------


#------ OrderBoor Class for Data Download--------------------------------------------------------------------------------
class OrderBook:
    def __init__(self, symbol:str, category:str = "spot", depth: str = 10, testnet: bool = True, desired_spread: int = 1):
        self.category = category
        self.symbol = symbol
        self.depth = depth
        self.testnet = testnet
        self.desired_spread = desired_spread
        self.AnalyticsData = pd.DataFrame()
        self.SpreadData = pd.DataFrame()
        
    def getOrderBookData(self, frames = 1, time_delay = 2):
        session = HTTP(testnet=self.testnet)
        data = session.get_orderbook(
                    category=self.category,
                    symbol= self.symbol,
                    limit=self.depth,
                    )

        FLAG = False
        frame = 0
        while frame < frames:
            if not FLAG:
                temp_df = pd.DataFrame(data['result']).rename(columns={'s': 'Asset', 'a': 'Ask', 'b': 'Bid', 'ts': 'Time', 'u': 'ID'})
                #--------ASK DATAFRAME ----------------------------------
                dfAsk = temp_df[['Asset', 'Ask', 'Time', 'ID']]
                dfAsk[['Price', 'Size']] = dfAsk['Ask'].apply(pd.Series)
                dfAsk['Side'] = "Ask"
                dfAsk.drop('Ask', axis=1, inplace=True)
                dfAsk['Size'] = pd.to_numeric(dfAsk['Size'])
                dfAsk['Price'] = pd.to_numeric(dfAsk['Price'])
                # :::::: Calculate for Ask ::::::::::::::::::::::::::::::
                minPriceAsk = dfAsk['Price'].min()
                maxPriceAsk = dfAsk['Price'].max()
                total_ask_size = dfAsk['Size'].sum()
                L1AskSize = float(data['result']['a'][0][-1]) # lowest ask price size
                total_ask_size_change = None
                #--------BIT DATAFRAME -----------------------------------
                dfBid = temp_df[['Asset', 'Bid', 'Time', 'ID']]
                dfBid[['Price', 'Size']] = dfBid['Bid'].apply(pd.Series)
                dfBid['Side'] = "Bid"
                dfBid.drop('Bid', axis=1, inplace=True)
                dfBid['Size'] = pd.to_numeric(dfBid['Size'])
                dfBid['Price'] = pd.to_numeric(dfBid['Price'])
                # :::::: Calculate for Bid :::::::::::::::::::::::::::::::
                minPriceBid = dfBid['Price'].min()
                maxPriceBid = dfBid['Price'].max()
                total_bid_size = dfBid['Size'].sum()
                L1BidSize = float(data['result']['a'][0][-1])
                total_bid_size_change = None


                # :::::: BOOK CALCULATION :::::::::::::::::::::::::::::::
                MidPrice = (maxPriceBid + minPriceAsk) / 2
                Spread = (maxPriceAsk - minPriceBid) / minPriceBid * 100
                TotalSize = total_ask_size + total_bid_size
                TotalSizeChange = None

                #======== Concat Ask and Bid Data and Populate Analytics Dataset ====================
                df = pd.concat([dfAsk, dfBid])
                df['MidPrice'] = float(MidPrice)
                df['Spread'] = float(Spread)
                df = df.sort_values('Price', ascending=True)

                new_row = {
                            'ID': df['ID'].iloc[0],
                            'Time': df['Time'].iloc[0],
                            'MidPrice': MidPrice,
                            'Spread': Spread,
                            'TotalSize': TotalSize,
                            'TotalSizeChange': TotalSizeChange,
                            'TotalBidSize': total_bid_size,
                            'TotalAskSize': total_ask_size,
                            'TotalBidSizeChange': None,  # Set appropriate value if needed
                            'TotalAskSizeChange': None,  # Set appropriate value if needed
                            'L1BitPrice': maxPriceBid,  
                            'L1AskPrice': minPriceAsk,  
                            'L1BitSize': L1BidSize,  
                            'L1AskSize': L1AskSize,  
                            'L1BitSizeChange': None ,  # Set appropriate value if needed
                            'L1AskSizeChange': None  # Set appropriate value if needed
                        }
                # self.AnalyticsData = self.AnalyticsData.append(new_row, ignore_index=True)
                self.AnalyticsData = pd.concat( [self.AnalyticsData, pd.DataFrame(new_row, index=[0])] )

                # ====== SPREAD DATA====================================================================
                # MaxPrice, MinPrice = self.SpreadMaxMinPrices(MidPrice, self.desired_spread)
                # temp_spread_data = df[df['Price'] > MinPrice | df['Price'] <= MaxPrice]


                FLAG = True
                frame += 1
            else:
                time.sleep(time_delay)
                data = session.get_orderbook(
                        category=self.category,
                        symbol= self.symbol,
                        limit=self.depth,
                        )

                temp_df = pd.DataFrame(data['result']).rename(columns={'s': 'Asset', 'a': 'Ask', 'b': 'Bid', 'ts': 'Time', 'u': 'ID'})

                if temp_df.ID.values[-1] == df.ID.values[-1]:
                    # print(f" IGNORING UPDATES ::::: \n REASON: REPEATING A FRAME OF OLD ID {df.ID.values[-1]} and  NEW ID {temp_df.ID.values[-1]}")
                    pass

                else:
                    dfAsk = temp_df[['Asset', 'Ask', 'Time', 'ID']]
                    dfAsk[['Price', 'Size']] = dfAsk['Ask'].apply(pd.Series)
                    dfAsk['Side'] = "Ask"
                    dfAsk.drop('Ask', axis=1, inplace=True)
                    dfAsk['Size'] = pd.to_numeric(dfAsk['Size'])
                    dfAsk['Price'] = pd.to_numeric(dfAsk['Price'])
                    # :::::: Calculate for Ask ::::::::::::::::::::::::::::::
                    minPriceAsk = dfAsk['Price'].min()
                    maxPriceAsk = dfAsk['Price'].max()
                    L1AskSizeChange = L1AskSize - float(data['result']['a'][0][-1])
                    L1AskSize = float(data['result']['a'][0][-1])
                    total_ask_size_change = total_ask_size - dfAsk['Size'].sum()
                    total_ask_size = dfAsk['Size'].sum()
                    #--------BIT DATAFRAME -----------------------------------
                    dfBid = temp_df[['Asset', 'Bid', 'Time', 'ID']]
                    dfBid[['Price', 'Size']] = dfBid['Bid'].apply(pd.Series)
                    dfBid['Side'] = "Bid"
                    dfBid.drop('Bid', axis=1, inplace=True)
                    dfBid['Size'] = pd.to_numeric(dfBid['Size'])
                    dfBid['Price'] = pd.to_numeric(dfBid['Price'])
                    # :::::: Calculate for Bid :::::::::::::::::::::::::::::::
                    minPriceBid = dfBid['Price'].min()
                    maxPriceBid = dfBid['Price'].max()
                    L1BidSizeChange = L1BidSize - float(data['result']['b'][0][-1])
                    L1BidSize = float(data['result']['b'][0][-1])
                    total_bid_size_change = total_bid_size - dfBid['Size'].sum()
                    total_bid_size = dfBid['Size'].sum()

                    # :::::: BOOK CALCULATION :::::::::::::::::::::::::::::::
                    MidPrice = (maxPriceBid + minPriceAsk) / 2
                    Spread = (maxPriceAsk - minPriceBid) / minPriceBid * 100
                    TotalSizeChange = TotalSize - (total_ask_size + total_bid_size)
                    TotalSize = total_ask_size + total_bid_size
                    
                    

                    #======== Concat Ask and Bid Data and Populate Analytics Dataset ====================
                    newdf = pd.concat([dfAsk, dfBid])
                    newdf = newdf.sort_values('Price', ascending=True)
                    newdf['MidPrice'] = float(MidPrice)
                    newdf['Spread'] = float(Spread)
                    newdf = newdf.sort_values('Price', ascending=True)
                    df = pd.concat([df, newdf])
                    
                    new_row = {
                            'ID': newdf['ID'].iloc[0],
                            'Time': newdf['Time'].iloc[0],
                            'MidPrice': MidPrice,
                            'Spread': Spread,
                            'TotalSize': TotalSize,
                            'TotalSizeChange': TotalSizeChange,
                            'TotalBidSize': total_bid_size,
                            'TotalAskSize': total_ask_size,
                            'TotalBidSizeChange': total_bid_size_change,  # Set appropriate value if needed
                            'TotalAskSizeChange': total_ask_size_change,  # Set appropriate value if needed
                            'L1BitPrice': maxPriceBid,  
                            'L1AskPrice': minPriceAsk,  
                            'L1BitSize': L1BidSize,  
                            'L1AskSize': L1AskSize,  
                            'L1BitSizeChange': L1BidSizeChange ,  # Set appropriate value if needed
                            'L1AskSizeChange': L1AskSizeChange  # Set appropriate value if needed
                        }

                    # self.AnalyticsData = self.AnalyticsData.append(new_row, ignore_index=True)
                    self.AnalyticsData = pd.concat( [self.AnalyticsData, pd.DataFrame(new_row, index=[0])] )
                    frame += 1
        
        self.AnalyticsData['Time'] = pd.to_datetime(self.AnalyticsData['Time'], unit='ms')
        df['Time'] = pd.to_datetime(df['Time'])
        print(f"The orderbook has {len(df.ID.unique())} snapshorts")
        return df
#-----------------------------------------------------------------------------------------------------------------------

#--------- Desired Spread Data Processing Class ------------------------------------------------------------------------
class SpreadData:
    def __init__(self, df: pd.DataFrame, percentage):

        self.df = df
        self.percentage = percentage
        self.SpreadData = pd.DataFrame()
        self.AggregateData = pd.DataFrame()

    def getData(self,):
        self.df['Size'] = pd.to_numeric(self.df['Size'])
        self.df['Price'] = pd.to_numeric(self.df['Price'])
        IDs = self.df.ID.unique()
        
        for i, ID in enumerate(IDs):
            ID_df = self.df[self.df['ID'] == ID]
            MidPrice = ID_df['MidPrice'].iloc[0]
            MaxPrice, MinPrice = self.SpreadMaxMinPrices(MidPrice)
            ID_df = ID_df[(ID_df['Price'] >= MinPrice) & (ID_df['Price'] <= MaxPrice)]

            # --------- Compute Stats ------------------
            if i > 0:
                TotalSizeChange = TotalSize - ID_df['Size'].sum()
                TotalBidSizeChange = TotalBidSize - ID_df[ID_df['Side'] == 'Bid']['Size'].sum() 
                TotalAskSizeChange = TotalAskSize - ID_df[ID_df['Side'] == 'Ask']['Size'].sum()

                BidPriceChange = BidPrice - ID_df[ID_df['Side'] == 'Bid']['Price'].mean()
                AskPriceChange = AskPrice - ID_df[ID_df['Side'] == 'Ask']['Price'].mean()
            else:
                BidPriceChange = None
                AskPriceChange = None
                TotalSizeChange = None
                TotalBidSizeChange = None
                TotalAskSizeChange = None

            TotalSize = ID_df['Size'].sum()
            TotalBidSize = ID_df[ID_df['Side'] == 'Bid']['Size'].sum()
            TotalAskSize = ID_df[ID_df['Side'] == 'Ask']['Size'].sum()

            BidPrice = ID_df[ID_df['Side'] == 'Bid']['Price'].mean()
            AskPrice = ID_df[ID_df['Side'] == 'Ask']['Price'].mean()

            if self.percentage > float(ID_df['Spread'].iloc[0]):
                Spread = float(ID_df['Spread'].iloc[0])
                SpredReduced = 0
            else:
                Spread = self.percentage
                SpredReduced = 1

            #========== populate Aggregate Data =====================
        
            new_row = {
                        'ID': ID_df['ID'].iloc[0],
                        'Time': ID_df['Time'].iloc[0],
                        'BidPrice': BidPrice,
                        'BidPriceChange': BidPriceChange,
                        'MidPrice': MidPrice,
                        'Spread': Spread,
                        'TotalSize': TotalSize,
                        'TotalSizeChange': TotalSizeChange,
                        'TotalBidSize': TotalBidSize,
                        'TotalBidSizeChange': TotalBidSizeChange,
                        'AskPrice': AskPrice,
                        'AskPriceChange': AskPriceChange,
                        'TotalAskSize': TotalAskSize,
                        'TotalAskSizeChange': TotalAskSizeChange,
                        'SpredReduced': SpredReduced
                    }
            # self.AnalyticsData = self.AnalyticsData.append(new_row, ignore_index=True)
            self.AggregateData = pd.concat( [self.AggregateData, pd.DataFrame(new_row, index=[0])] )
            self.SpreadData = pd.concat( [self.SpreadData, ID_df] )
            self.SpreadData['Spread'] = Spread
            self.SpreadData['MidPrice'] = MidPrice


            
    def SpreadMaxMinPrices(self, SpreadMidPrice):

        SpreadMidPrice = float(SpreadMidPrice)
        percentage = float(SpreadMidPrice) * (self.percentage/100)
        MaxPrice = SpreadMidPrice + percentage
        MinPrice = SpreadMidPrice - percentage
    
        return MaxPrice, MinPrice

#------------ Binning OrderBooK Price Functions -----------------------------------

def BinPrices(data: pd.DataFrame, bins = 50, significant_fig = 6, bounds: list = []):
    df = data.copy(deep=True)
    # Calculate the bin edges
    df['Price'] = pd.to_numeric(df['Price'])
    df['Size'] = pd.to_numeric(df['Size'])
    # Perform binarization
    bin_edges = np.histogram_bin_edges(df['Price'], bins=bins+1)
    df['Bin'] = np.digitize(df['Price'], bin_edges)

    # Convert the values to bins
    df['bin_price'] = df.groupby('Bin')['Price'].transform('mean')
    df['bin_size'] = df.groupby(['ID','Bin'])['Size'].transform('sum')
    df['price_count'] = df.groupby(['ID','Bin'])['Size'].transform('count')
    df['bin_price'] = df['bin_price'].round(significant_fig).astype(str)
    df['Price'] = df['Price'].round(significant_fig).astype(str)
    df.drop_duplicates(subset=['ID','bin_price', 'Side'], inplace=True)

    return df

#------------------------------------------------------------------------------------

