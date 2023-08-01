import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tikzplotlib
import math
from scipy import stats
from datetime import date
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

# To use the experimental IterativeImputer, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from scipy.stats.mstats import winsorize

def dataSequence(yearBuildBuckets=5,coordinatBucketSize=5,kmeansCluster=0,is_age_bucket=True,filterRent=False,imputationArea=False, winsor=0):
    data = loadStivadData(imputedArea=imputationArea)

    if filterRent : data = filterNonRent(data)
    data = dataPrepStatic(data)
    data = dataPrepDynamic(data=data,yearBuildBuckets=yearBuildBuckets,coordinatBucketSize=coordinatBucketSize, is_age_bucket=is_age_bucket)

    usekmeans = kmeansCluster > 0
    if usekmeans : data = clusterLocationOnKMM(data,kmeansCluster)
    if imputationArea : data = imputArea(data, usekmeans)

    data = calculations(data, winsor)
    data = addEnergyPriceData(data)
    return data

def addEnergyPriceData(df):
    energyPrices = pd.read_csv('./data/20230720_energyPrices.csv', delimiter=',')
    energyPrices["day"] = 1
    energyPrices["date"] = pd.to_datetime(pd.to_datetime(dict(year=energyPrices.jaar, month=energyPrices.maand, day=energyPrices.day)))
    transposed = energyPrices.set_index(energyPrices["date"])[["electricity","gas"]].T
    transposed = transposed.groupby(pd.PeriodIndex(transposed.columns, freq='Q'), axis=1).mean().T.reset_index()
    df['qtr'] = pd.PeriodIndex(pd.to_datetime(df.transaction_agreement_date), freq='Q')
    return pd.merge(df, transposed, how="left", left_on="qtr", right_on='date')


def clusterLocationOnKMM(df,kmeansCluster=400):
    cities = pd.read_csv('./data/independVars/20230716_dutchCities.csv', sep=",")
    X = df[["address_latitude","address_longitude","address_city"]]
    X = X.merge(cities, left_on="address_city", right_on="city",how="left")

    cities = X[["address_latitude","address_longitude","admin_name"]]
    cities = pd.get_dummies(cities, columns=["admin_name"], drop_first=False)
    
    kmeans = KMeans(
        init="random",
        n_clusters=kmeansCluster,
        n_init=10,
        max_iter=20000,
        random_state=42
    )
    
    df["kmeans_cluster"] = kmeans.fit(cities).labels_
    
    return df

def imputArea(df, useKmeans):

    cat = ["property_property_type","transaction_transaction_type","building_age_at_transaction_bucket","transaction_company_type","transaction_counter_party_type","property_energy_label","mixedUseDummy","transaction_year"]
    con = ["property_construction_year","calculations_sum_units","calculations_total_purchase_costs","purchase_purchase_price","purchase_buyer_cost","calculations_sum_rent_revision","calculations_operating_costs","calculations_sum_rent","calculations_sum_rent_revision_inc_corr"]
    y_name = ["calculations_sum_area"]

    if useKmeans : 
        print(useKmeans)
        cat += ["kmeans_cluster"]
    
    df["un_imputed"] = df["calculations_sum_area"]

    fullData = df.copy()
    fullData = fullData[y_name + cat + con]
    fullData["calculations_sum_area"] = fullData["calculations_sum_area"].replace(0, np.NaN)
    fullData = pd.get_dummies(fullData, columns=cat, drop_first=True)

    imp = IterativeImputer(max_iter=100, random_state=0, verbose=2)
    df["calculations_sum_area"] = imp.fit_transform(fullData)[:,0]

    return df

def dataPrepStatic(data):
    data = dateFormatingPrep(data)
    data = energyLabelPrep(data)
    data = propertyCategoryPrep(data)
    data = mixedUseDummy(data)
    # data = mergeWithPC4Default(data)
    data = data.rename(columns={"CBSpostcode4":'pc4','CBSgemiddelde_woz_waarde_woning':'woz','CBSstedelijkheid':'stedelijkheid'})
    data = findCostOfCapital(data)
    data = isInRandstad(data)
    data = simplifyTransactions(data)
    return data

def simplifyTransactions(df):
    transactionDict = dict({
        "sale":"sale",
        "turnkey":"construction",
        "construction":"construction",
        "purchase":"purchase",
        "redevelopment-buy":"construction",
        "buy-own-use":"sale",
        "sale-lease-back":"purchase",
    })
    df["transactions_simplified"] = df["transaction_transaction_type"].replace(transactionDict)
    return df

def dataPrepDynamic(data,yearBuildBuckets=15,coordinatBucketSize=2, is_age_bucket=False):
    data = createLocationBuckets(data,coordinatBucketSize)
    if is_age_bucket : createAgeBucket(data,yearBuildBuckets)
    else : data = createYearBuckets(data,yearBuildBuckets)
    return data


def isInRandstad(df):
    top150k = [ 
        "Amsterdam",
        "Rotterdam",
        "Den Haag",
        "Utrecht",
        "Eindhoven",
        "Groningen",
        "Tilburg",
        "Almere",
        "Breda",
        "Nijmegen",
        "Apeldoorn",
        "Arnhem",
        "Haarlem",
        "Haarlemmermeer",
        "Enschede",
        "Amersfoort",
        "Zaanstad",
        "'s-Hertogenbosch"
    ]
    randStad = [
        "Amsterdam",
        "Rotterdam",
        "Den Haag",
        's-Gravenhage',
        "Utrecht",
        "Almere",
        "Haarlem",
        "Amersfoort",
        "Zaanstad",
        "Haarlemmermeer",
        "Zoetermeer",
        "Leiden",
        "Dordrecht",
        "Alkmaar",
        "Delft"
    ]
    biggestCities = [
        "Amsterdam",
        "Rotterdam",
        "Den Haag",
        's-Gravenhage'
        "Utrecht",
    ]
    df["is_in_randstad"] = df["address_city"].isin(randStad)
    df["is_in_top150k"] = df["address_city"].isin(top150k)
    df["is_Amsterdam"] = df["address_city"] == "Amsterdam"
    df["is_in_biggestCities"] = df["address_city"].isin(biggestCities)
    return df

def findMixedUse(dfRow):
    expoTypes = ["exploitation_housing_total_area","exploitation_housing_care_total_area","exploitation_shop_total_area","exploitation_office_total_area","exploitation_industrial_total_area","exploitation_other_total_area"]
    # exploitations = dfRow[expoTypes]
    # if (len(expoTypes) - dfRow[expoTypes].isna().sum().sum()) > 1 : return True
    return (len(expoTypes) - dfRow[expoTypes].isna().sum().sum()) > 1


def mixedUseDummy(df):
    df["mixedUseDummy"] = df.apply(lambda x: findMixedUse(x), axis=1)
    return df

def findCostOfCapital(df,inflationTarget=0.02,INDEPENVARSDIR='./data/independVars/',fileName='20230621_costOfCapital.csv'):
    costSet = pd.read_csv(INDEPENVARSDIR+fileName)
    df["fundamental_value"] = df.apply(lambda x: calculateFundamentalValue(x, costSet, inflationTarget), axis=1)
    df["difference_fundamentalValue_transaction"] = df["purchase_purchase_price"] - df["fundamental_value"]
    df["difference_fundamentalValue_transaction_percent"] = 100 / df["purchase_purchase_price"] * df["difference_fundamentalValue_transaction"]
    return df

def calculateFundamentalValue(dfrow, costSet, inflationTarget):
    wacc = costSet[costSet["year"] == dfrow["transaction_year"]]["localCurrInterpolation"]
    if len(wacc) > 0:
        return float(dfrow["calculations_sum_rent"] / (wacc.iloc[0]-inflationTarget))
    return np.nan

def calculations(df, winsor):
    df["purchase_price_log"] = np.log(df["purchase_purchase_price"])
    
    # Operating costs
    df["calculations_operating_costs_log"] = df["calculations_operating_costs"]
    df.loc[df["calculations_operating_costs_log"]<=0,"calculations_operating_costs_log"] = 1
    df["calculations_operating_costs_log"] = np.log(df["calculations_operating_costs_log"])
    
    # Sum area
    df["calculations_sum_area_log"] = df["calculations_sum_area"]
    df.loc[df["calculations_sum_area_log"]<=0,"calculations_sum_area_log"] = 1
    df["calculations_sum_area_log"] = np.log(df["calculations_sum_area_log"])
    
    # Building age
    df["building_age_at_transaction_log"] = df["building_age_at_transaction"]
    df.loc[df["building_age_at_transaction_log"]<=0,"building_age_at_transaction_log"] = 1
    df["building_age_at_transaction_log"] = np.log(df["building_age_at_transaction_log"])
    df["building_age_at_transaction_squared"] = np.square(df["building_age_at_transaction"])
    df["building_age_at_transaction_cubed"] = df["building_age_at_transaction"]**3
    df["difference_area_leased"] = df["calculations_sum_area"] - df["calculations_sum_area_leased"]
    
    # Price per meter
    df["price_per_meter2"] = df.loc[:,"purchase_purchase_price"] / df.loc[:,"calculations_sum_area"]
    
    if winsor != 0:
        df["price_per_meter2_untouched"] = df["price_per_meter2"]
        df["price_per_meter2"] = winsorize(df["price_per_meter2"],(winsor))

    df["price_per_meter2_z"] = np.abs(stats.zscore(df["price_per_meter2"]))
    df["price_per_meter2_log"] = np.log2(df["price_per_meter2"])
    
    # Rent
    df["rent_per_m2"] = df.loc[:,"calculations_sum_rent_th"] / df.loc[:,"calculations_sum_area"]
    df["rent_per_m2_z"] = np.abs(stats.zscore(df["rent_per_m2"]))
    df["rent_per_m2_log"] = df["rent_per_m2"]
    df.loc[df["rent_per_m2_log"]<=0,"rent_per_m2_log"] = 1
    df["rent_per_m2_log"] = np.log2(df["rent_per_m2_log"])
    
    df["cap_rate"] = 100 / df["price_per_meter2"] * df["rent_per_m2"]
    
    df["renovated"] = (df["property_last_renovation_year"] > 1800) & (df["property_last_renovation_year"].fillna(0).astype(int) != df["property_construction_year"].fillna(0).astype(int))
    # df["renovated_yearsAgo"] =  df.apply(lambda x: returnRenovationYearsAgo(x), axis=1)
    # (df["property_last_renovation_year"] > 1800) & (df["property_last_renovation_year"].fillna(0).astype(int) != df["property_construction_year"].fillna(0).astype(int))
    
    return df

# def returnRenovationYearsAgo(dfRow):
#     if dfRow["property_last_renovation_year"] > 1800 and dfRow["property_last_renovation_year"].fillna(0).astype(int) != dfRow["property_construction_year"].fillna(0).astype(int):
#         return int(dfRow["transaction_year"] - dfRow["property_last_renovation_year")]
#     return np.Nan()

def propertyCategoryPrep(df):
    df["property_property_type_categorized"] = df["property_property_type"]
    df["property_property_type_categorized"] = df["property_property_type_categorized"].astype("category")
    df["property_property_type_categorized"] = df["property_property_type_categorized"].cat.codes
    return df

def energyLabelPrep(df):
    # df["categorizedEnergyLabel"] = df["property_energy_label"].fillna('na-c')
    df["categorizedEnergyLabel"] = df["property_energy_label"].fillna('na')
    
    df["categorizedEnergyLabel_simple"] = df["categorizedEnergyLabel"].replace(['a','aa','aaa','aaaa','aaaaa','aaaaaa'],'a')
    df["categorizedEnergyLabel"] = df["categorizedEnergyLabel"].astype("category")
    df["categorizedEnergyLabel"] = df["categorizedEnergyLabel"].cat.reorder_categories(['na', 'g', 'f', 'e', 'd','c','b','a','aa','aaa','aaaa','aaaaa','aaaaaa'], ordered=True)
    df["categorizedEnergyLabel"] = df["categorizedEnergyLabel"].cat.codes
    df["is_efficient"] = (df["categorizedEnergyLabel_simple"] == 'a') | (df["categorizedEnergyLabel_simple"] == 'b') | (df["categorizedEnergyLabel_simple"] == 'c')
    df["is_above_b"] = (df["categorizedEnergyLabel_simple"] == 'a') | (df["categorizedEnergyLabel_simple"] == 'b')
    # df["categorizedEnergyLabel_simple_suplemended"] = df["categorizedEnergyLabel_simple"]
    
    # df["categorizedEnergyLabel_simple_suplemended"] = df.apply(lambda x: returnConstructionVariable(x), axis=1)
    df["categorizedEnergyLabel_simple_suplemended"] = df["categorizedEnergyLabel_simple"]
    df = suplementEnergyLabels(df)
    # df["categorizedEnergyLabel_simple_cominedNA"] = df["categorizedEnergyLabel_simple"].replace('na-c','na')
    return df

def returnConstructionVariable(df_row):
    if df_row["categorizedEnergyLabel_simple"] != 'na' : return df_row["categorizedEnergyLabel_simple"]
    # if df_row["property_construction_year"] >= 2023 : return 'construction'
    return 'na'

def suplementEnergyLabels(df):
    
    df.loc[df["transaction_transaction_id"] == 2366,"categorizedEnergyLabel_simple_suplemended"] = 'd' # https://kadastralekaart.com/adres/utrecht-steenweg-34/0344200000147478
    df.loc[df["transaction_transaction_id"] == 520,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/leidschendam-js-bachlaan-3/1916200000015802
    df.loc[df["transaction_transaction_id"] == 548,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/leidschendam-liguster-31/1916200000019499
    df.loc[df["transaction_transaction_id"] == 767,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/amsterdam-nieuwe-passeerdersstraat-2/0363200012084468
    df.loc[df["transaction_transaction_id"] == 790,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/s-hertogenbosch-hooge-steenweg-25/0796200000435910
    df.loc[df["transaction_transaction_id"] == 790,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/s-hertogenbosch-hooge-steenweg-25/0796200000435910
    df.loc[df["transaction_transaction_id"] == 2245,"categorizedEnergyLabel_simple_suplemended"] = 'c' # https://kadastralekaart.com/adres/maastricht-kleine-staat-10/0935200000002071
    df.loc[df["transaction_transaction_id"] == 416,"categorizedEnergyLabel_simple_suplemended"] = 'a' #https://kadastralekaart.com/adres/s-hertogenbosch-schapenmarkt-14/0796200000433238
    df.loc[df["transaction_transaction_id"] == 2154,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://www.premiersuiteseurope.com/nl/blog/the-hourglass-project
    df.loc[df["transaction_transaction_id"] == 2975,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://www.funda.nl/huur/amsterdam/appartement-42042807-hugo-de-vrieslaan-1-m/
    df.loc[df["transaction_transaction_id"] == 2485,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/amsterdam-gaasterlandstraat-8/0363200012102757
    df.loc[df["transaction_transaction_id"] == 1075,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/arnhem-willemsplein-39/0202200000399376
    df.loc[df["transaction_transaction_id"] == 2974,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://www.wonenbijbouwinvest.nl/huuraanbod/1/rachmaninoffhuis
    df.loc[df["transaction_transaction_id"] == 411,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/vleuten-herfsttuinlaan-10/0344200000158981
    df.loc[df["transaction_transaction_id"] == 2861,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/amersfoort-prins-frederiklaan-4/0307200000406194
    df.loc[df["transaction_transaction_id"] == 2277,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/amsterdam-wibautstraat-80/0363200000419192
    df.loc[df["transaction_transaction_id"] == 472,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://drimble.nl/adres/amsterdam/1011DJ/simon-carmiggeltstraat-50.html
    df.loc[df["transaction_transaction_id"] == 2646,"categorizedEnergyLabel_simple_suplemended"] = 'a' #https://kadastralekaart.com/adres/amsterdam-welnastraat-1/0363200012066052
    df.loc[df["transaction_transaction_id"] == 2390,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/amsterdam-eef-kamerbeekstraat-630/0363200012120553
    df.loc[df["transaction_transaction_id"] == 3430,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/amsterdam-tt-vasumweg-32/0363200012141524
    df.loc[df["transaction_transaction_id"] == 2050,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/amsterdam-eef-kamerbeekstraat-1/0363200012095709
    df.loc[df["transaction_transaction_id"] == 2550,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/amsterdam-revaleiland-3/0363200012135371
    df.loc[df["transaction_transaction_id"] == 2490,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/amsterdam-revaleiland-3/0363200012135371
    df.loc[df["transaction_transaction_id"] == 2451,"categorizedEnergyLabel_simple_suplemended"] = 'b' # https://kadastralekaart.com/adres/amsterdam-revaleiland-3/0363200012135371
    df.loc[df["transaction_transaction_id"] == 2647,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/amsterdam-bella-vistastraat-2/0363200012124636
    df.loc[df["transaction_transaction_id"] == 2241,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://www.amsterdamwoont.nl/actueel/xavier-stijlvolle-huurappartementen-op-de-zuidas-vanaf-1-240-euro-per-maand-inschrijven-t-m-1-februari
    df.loc[df["transaction_transaction_id"] == 2640,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://bezuidenhout.nl/wp-content/uploads/2016/11/RIS296560_bijlage_schetsontwerp_Grotiusplaats.pdf
    df.loc[df["transaction_transaction_id"] == 2254,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://cu2030.nl/project/het-platform
    df.loc[df["transaction_transaction_id"] == 2631,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://vorm.nl/actueel/nieuws/93-koop-en-huurappartementen-opgeleverd-in-de-stadhouders-den-haag
    df.loc[df["transaction_transaction_id"] == 720,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://vorm.nl/actueel/nieuws/93-koop-en-huurappartementen-opgeleverd-in-de-stadhouders-den-haag
    df.loc[df["transaction_transaction_id"] == 2061,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/amsterdam-bert-haanstrakade-800/0363200012115788
    df.loc[df["transaction_transaction_id"] == 2064,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/amsterdam-bert-haanstrakade-800/0363200012115788
    df.loc[df["transaction_transaction_id"] == 2064,"categorizedEnergyLabel_simple_suplemended"] = 'a' # https://kadastralekaart.com/adres/amsterdam-bert-haanstrakade-800/0363200012115788
    
    # df.loc[df["transaction_transaction_id"] == 1149,"categorizedEnergyLabel_simple_suplemended"] = 'e' # https://kadastralekaart.com/adres/haarlem-grote-houtstraat-80/0392200000022519

    return df

def buyerSellerCat(df):
    companyTypeDict = {
        'institutional_investor':0,
        'other':1,
        'social_houser':2,
        'private':3,
        'foreign_investor':4,
        'project_developer':5,
        'government':6,
        'contractor':7,
        'owner':8
    }
    df["transaction_seller_categorized"] = df["transaction_counter_party_type"].replace(companyTypeDict).astype("category")
    df["transaction_buyer_categorized"] = df["transaction_company_type"].replace(companyTypeDict).astype("category")
    return df

def dateFormatingPrep(df):
    yearFirst = "%Y/%m/%d"
    dayFirst = "%d/%m/%Y"
    monthFirst = "%m/%d/%Y"

    df["transaction_agreement_date_formated"] = pd.to_datetime(df['transaction_agreement_date'].astype(str), format=yearFirst)
    df["building_age_at_transaction"] = df["transaction_agreement_date_formated"].dt.year - df["property_construction_year"]
    df["transaction_year"] = df["transaction_agreement_date_formated"].dt.year
    df["transaction_quater"] = pd.PeriodIndex(df.transaction_agreement_date_formated, freq='Q')
    df["transaction_quater_categorized"] = df["transaction_quater"].astype(str)
    
    # Lag
    df["lag_3m"] = pd.to_datetime(df["transaction_agreement_date_formated"]) + pd.DateOffset(months=-3)
    df["lag_6m"] = pd.to_datetime(df["transaction_agreement_date_formated"]) + pd.DateOffset(months=-6)

    # Transaction moment booleans
    df["transaction_agreed_after_BouwBesluit"] = df["transaction_agreement_date_formated"] > "2011-08-29"
    df["transaction_afterCovid"] = df["transaction_agreement_date_formated"] > "2020-03-01" 

    df["transaction_afterInterestHike"] = df["transaction_agreement_date_formated"] > "2022-07-27"
    df["transaction_l3_after_First_InterestHike"] = df["lag_3m"] > "2022-07-27"
    df["transaction_l9_after_First_InterestHike"] = df["lag_6m"]  > "2022-07-27"
    df["transaction_l3_after_second_InterestHike"] = df["lag_3m"] > "2022-09-14"
    df["transaction_l9_after_second_InterestHike"] = df["lag_6m"]  > "2022-09-14"

    interestRates = pd.read_csv('./data/interestRate.csv')
    interestRates.transaction_agreement_date_formated = pd.to_datetime(interestRates.transaction_agreement_date_formated)
    df = df.merge(interestRates, on="transaction_agreement_date_formated", how="left")

    return df


def createAgeBucket(df, bucketSize= 5):
    df["building_age_non_codes"] = pd.qcut(df["building_age_at_transaction"], bucketSize)
    df["building_age_at_transaction_bucket"] = df["building_age_non_codes"].cat.codes
    return df

def createYearBuckets(df, bucketSize = 15):
    df["yearBuildBucket_bucket"] = pd.qcut(df["property_construction_year"], bucketSize).cat.codes
    return df

def createLocationBuckets(df, bucketSize = 2):
    latMax = df.address_latitude.max()
    latMin = df.address_latitude.min()
    longMax = df.address_longitude.max()
    longMin = df.address_longitude.min()
    Yaxis = getDistanceFromLatLonInKm(latMin,longMin,latMax,longMin)
    Xaxis = getDistanceFromLatLonInKm(latMin,longMin,latMin,longMax)
    SquareSize = bucketSize
    LongRange = np.linspace(longMin,longMax,int(np.ceil(Xaxis/SquareSize)))
    LatRange = np.linspace(latMin,latMax,int(np.ceil(Yaxis/SquareSize)))
    df["longBucket"] = pd.cut(df.address_longitude, LongRange).cat.codes
    df["latBucket"] = pd.cut(df.address_latitude, LatRange).cat.codes
    df["coordinatBucket"] = df["longBucket"].astype(str) + ';' + df["latBucket"].astype(str)
    df["coordinatBucket_category"] = df["coordinatBucket"].astype("category").cat.codes
    return df

def mergeWithPC4Default(df,INDEPENVARSDIR='./data/independVars/',PC4ENERGY='energielabel_naar_pc4_cre.dta'):
    pc4 = pd.read_stata(INDEPENVARSDIR+PC4ENERGY)
    pc4 = pc4.rename(columns={"postcode": "pc4", "label":"pc4_energyLabel"})
    df = df.rename(columns={"CBSpostcode4":'pc4'}) ##["pc4"] = df["address_postal_code"].str[:4]
    df = df.merge(pc4, on="pc4", how="left")
    df["categorizedEnergyLabel_simple_extended"] = df.apply(lambda x: find_pc4_defaultValue(x), axis=1)

    df["categorizedEnergyLabel_simple_extended"] = df["categorizedEnergyLabel_simple_extended"].str.lower()
    
    return df

def find_pc4_defaultValue(pandasRow):
    if pandasRow["categorizedEnergyLabel_simple"] != 'na':
        return pandasRow["categorizedEnergyLabel_simple"]
    if len(str(pandasRow["address_postal_code"])) == 0 or pandasRow["address_postal_code"] == '':
        return 'na'
    return pandasRow["pc4_energyLabel"]

def filterNonRent(df):
    df = df[df["calculations_sum_rent_th"] != 0]
    return df


def loadStivadData(DATADIR='./data/',INDEPENVARSDIR='./data/independVars/',DATAVERSION='20230601',STIVADEXPORT='_stivad-export.csv',CBSEXPORT='pc4_join.csv',imputedArea=False):
    stivad_raw = pd.read_csv(DATADIR+DATAVERSION+STIVADEXPORT, delimiter=",")
    
    cbs = pd.read_csv(DATADIR+CBSEXPORT, delimiter=",")
    cbs = cbs.replace(-99997, np.nan)

    stivad_raw = stivad_raw.merge(cbs, on="transaction_transaction_id", how="left")
    
    stivad_raw = mutateSize(stivad_raw)
    stivad_raw = deleteRows(stivad_raw)
    stivad_raw = stivad_raw[stivad_raw["property_property_type"] != 'parking']
    stivad_raw = stivad_raw[stivad_raw["transaction_object_part"] == 'whole'] # TODO: Find a way to include partial buildings
    if imputedArea:
        print('Do not filter')
    else:
        stivad_raw = stivad_raw[stivad_raw.calculations_sum_area != 0]
    stivad_raw = stivad_raw[stivad_raw.purchase_purchase_price != 0]
    return stivad_raw

def deleteRows(df):
    # missing reliable information on area size
    df = df[df.transaction_transaction_id != 715] # Toren 1 & 2 "De Reigers" -> Onbekende oppervlakte woondelen
    df = df[df.transaction_transaction_id != 713] # Toren 3 de reigers 
    df = df[df.transaction_transaction_id != 749] # Lamgroen / spui -> ?
    df = df[df.transaction_transaction_id != 2471] # Frites attelier -> Hebben ze nu echt meer dan een miljoen betaald voor 25m2?
    df = df[df.transaction_transaction_id != 806] # van der Heydenstraat Zwolle -> project is vermoedelijk geklapt?

    df = df[df.transaction_transaction_id != 745] # Nieuwehaven 1; een heel raar project
    df = df[df.transaction_transaction_id != 745] # Nieuwehaven 1; een heel raar project

    # df = df[df.transaction_transaction_id != 7689] # Outliers
    # df = df[df.transaction_transaction_id != 6441] # Outliers
    # df = df[df.transaction_transaction_id != 4449] # Outliers
    # df = df[df.transaction_transaction_id != 3476] # Outliers
    # df = df[df.transaction_transaction_id != 3074] # Outliers
    # df = df[df.transaction_transaction_id != 2651] # Outliers
    # df = df[df.transaction_transaction_id != 2127] # Outliers
    # df = df[df.transaction_transaction_id != 1815] # Outliers
    # df = df[df.transaction_transaction_id != 1654] # Outliers
    # df = df[df.transaction_transaction_id != 1274] # Outliers
    # df = df[df.transaction_transaction_id != 1018] # Outliers
    # df = df[df.transaction_transaction_id != 976]  # Outliers
    # df = df[df.transaction_transaction_id != 863]  # Outliers
    # df = df[df.transaction_transaction_id != 695]  # Outliers
    # df = df[df.transaction_transaction_id != 477]  # Outliers
    # df = df[df.transaction_transaction_id != 478]  # Outliers
    # df = df[df.transaction_transaction_id != 446]  # Outliers
    # df = df[df.transaction_transaction_id != 1321]  # Outliers
    # df = df[df.transaction_transaction_id != 1627]  # Outliers
    # df = df[df.transaction_transaction_id != 374]  # Outliers
    # df = df[df.transaction_transaction_id != 989]  # Outliers
    # df = df[df.transaction_transaction_id != 1056]  # Outliers

    df = df[df.transaction_transaction_id != 465]  # Molenberg 22 Den Bosch gaat om 26 units, maar er is maar 64m2 opgegeven
    df = df[df.transaction_transaction_id != 401]  # Molenberg 22 Den Bosch gaat om 26 units, maar er is maar 64m2 opgegeven

    return df

def mutateSize(df):
    df.loc[df["transaction_transaction_id"] == 720,"calculations_sum_area"] = 3558 # https://gprprojecten.nl/Home/Details/51200/Kernhem-Scherf-1
    df.loc[df["transaction_transaction_id"] == 9887,"calculations_sum_area"] = 2037 # Totaal metrage 2.037 m2
    df.loc[df["transaction_transaction_id"] == 9881,"calculations_sum_area"] = 1308 # totaal oppervlak van de 12 woningen: 1.308 m2
    df.loc[df["transaction_transaction_id"] == 465,"calculations_sum_area"] = 65 * 26
    df.loc[df["transaction_transaction_id"] == 401,"calculations_sum_area"] = 65 * 26
    return df

def getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2):
  R = 6371
  dLat = deg2rad(lat2-lat1)
  dLon = deg2rad(lon2-lon1) 
  a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
  d = R * c
  return d


def deg2rad(deg):
  return deg * (math.pi/180)

nameDict = dict({
    "C(is_efficient)[T.True]"                                                           :   'Is efficient',
    "C(categorizedEnergyLabel_simple_suplemended, Treatment(reference='d'))[T.a]"       :   'A',
    "C(categorizedEnergyLabel_simple_suplemended, Treatment(reference='d'))[T.b]"       :   'B',
    "C(categorizedEnergyLabel_simple_suplemended, Treatment(reference='d'))[T.c]"       :   'C',
    "C(categorizedEnergyLabel_simple_suplemended, Treatment(reference='d'))[T.e]"       :   'E',
    "C(categorizedEnergyLabel_simple_suplemended, Treatment(reference='d'))[T.f]"       :   'F',
    "C(categorizedEnergyLabel_simple_suplemended, Treatment(reference='d'))[T.g]"       :   'G',
    "C(categorizedEnergyLabel_simple_suplemended, Treatment(reference='d'))[T.na]"      :   'NA',
    "C(categorizedEnergyLabel_simple_suplemended, Treatment(reference='d'))[T.na-c]"    :   'NA-C',
    'C(building_age_at_transaction_bucket, Treatment(reference=0))[T.1]'                :'age bin: (-2.0, -1.0]',
    'C(building_age_at_transaction_bucket, Treatment(reference=0))[T.2]'                :'age bin: (-1.0, 0.0]',
    'C(building_age_at_transaction_bucket, Treatment(reference=0))[T.3]'                :'age bin: (0.0, 9.0]',
    'C(building_age_at_transaction_bucket, Treatment(reference=0))[T.4]'                :'age bin: (9.0, 17.0]',
    'C(building_age_at_transaction_bucket, Treatment(reference=0))[T.5]'                :'age bin: (17.0, 23.0]',
    'C(building_age_at_transaction_bucket, Treatment(reference=0))[T.6]'                :'age bin: (23.0, 29.0]',
    'C(building_age_at_transaction_bucket, Treatment(reference=0))[T.7]'                :'age bin: (29.0, 41.0]',
    'C(building_age_at_transaction_bucket, Treatment(reference=0))[T.8]'                :'age bin: (41.0, 68.3]',
    'C(building_age_at_transaction_bucket, Treatment(reference=0))[T.9]'                :'age bin: (68.3, 418.0]',
    'C(mixedUseDummy)[T.True]'                                                          :'Mixed use',
    'C(property_land_ownership)[T.owner]'                                               :'Land ownership: Owner',
    'C(property_land_ownership)[T.perpetual_lease]'                                     :'Land ownership: Perpetual lease',
    'C(property_land_ownership)[T.prepaid_lease]'                                       :'Land ownership: prepaid lease',
    'C(property_property_type)[T.housing_care]'                                         :'Property type: Housing care',
    'C(property_property_type)[T.industrial]'                                           :'Property type: Industrial',
    'C(property_property_type)[T.office]'                                               :'Property type: Office',
    'C(property_property_type)[T.other]'                                                :'Property type: Other',
    'C(property_property_type)[T.shop]'                                                 :'Property type: Shop',
    'C(renovated)[T.True]'                                                              :'Renovated',
    'C(transaction_year, Treatment(reference=2017))[T.2008]'                            :'Year: 2008',
    'C(transaction_year, Treatment(reference=2017))[T.2009]'                            :'Year: 2009',
    'C(transaction_year, Treatment(reference=2017))[T.2010]'                            :'Year: 2010',
    'C(transaction_year, Treatment(reference=2017))[T.2011]'                            :'Year: 2011',
    'C(transaction_year, Treatment(reference=2017))[T.2012]'                            :'Year: 2012',
    'C(transaction_year, Treatment(reference=2017))[T.2013]'                            :'Year: 2013',
    'C(transaction_year, Treatment(reference=2017))[T.2014]'                            :'Year: 2014',
    'C(transaction_year, Treatment(reference=2017))[T.2015]'                            :'Year: 2015',
    'C(transaction_year, Treatment(reference=2017))[T.2016]'                            :'Year: 2016',
    'C(transaction_year, Treatment(reference=2017))[T.2018]'                            :'Year: 2018',
    'C(transaction_year, Treatment(reference=2017))[T.2019]'                            :'Year: 2019',
    'C(transaction_year, Treatment(reference=2017))[T.2020]'                            :'Year: 2010',
    'C(transaction_year, Treatment(reference=2017))[T.2021]'                            :'Year: 2021',
    'C(transaction_year, Treatment(reference=2017))[T.2022]'                            :'Year: 2022',
    'C(transaction_year, Treatment(reference=2017))[T.2023]'                            :'Year: 2023',
    'C(transactions_simplified)[T.purchase]'                                            :'Transaction type: Purchase',
    'C(transactions_simplified)[T.sale]'                                                :'Transaction type: Sale',
    "C(is_efficient)[T.True]:C(transactions_simplified)[T.purchase]"                     :'Efficient * Purchase',
    "C(is_efficient)[T.True]:C(transactions_simplified)[T.sale]"                         :'Efficient * Sale',
    "C(is_Amsterdam)[T.True]":"Amsterdam",
    "C(is_efficient)[T.True]:C(is_Amsterdam)[T.True]":"Efficient * Amsterdam",
    "C(is_in_top150k)[T.True]" : "Cities with over 150k citizens",
    "C(is_efficient)[T.True]:C(is_in_top150k)[T.True]": "Efficient * Cities with over 150k citizens",
    "C(is_efficient)[T.True]:C(property_property_type)[T.housing]" : "Efficient * housing",
    "C(is_efficient)[T.True]:C(property_property_type)[T.shop]" : "Efficient * shop",
    "C(is_efficient)[T.True]:C(property_property_type)[T.office]" : "Efficient * office",
    })