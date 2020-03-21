#!/usr/bin/env python
# coding: utf-8

# # Pitch Data - Web Scrapping and Database creation
# 

# In[ ]:


import pandas as pd
import os.path
import requests
import csv
import pymysql
from bs4 import BeautifulSoup


# <center><b><font size="4">Web Scraping</font></b></center>
# 

# Scrape the Batter_Info

# In[ ]:


## Step 1: Creating a function to scrape Starting Batters Data
def getStandardBatting(year):
    csvfile = str(year) + '-standard-batting.csv'
    if os.path.isfile(csvfile):
        dd = pd.read_csv(csvfile)
    else:
        #request the different URLS based on function input
        url = requests.get("https://www.baseball-reference.com/leagues/MLB/"+str(year)+"-standard-batting.shtml")
        #pull the HTML code
        soup = BeautifulSoup(url.content, "html.parser")
        #Find the table we want to scrap
        div = soup.find('div', id='all_players_standard_batting')
        #find the contents we want to scrap
        bdiv = bytearray(str(div.contents),'utf-8')
        #pull that contents into a soup object 
        soup2 = BeautifulSoup(bdiv, "html.parser")
        tbody = soup2.find('tbody')
        #find all the rows we want
        rows = tbody.find_all('tr')
        #label all the columns
        columns = ['Name','Age','Tm']
        #makes these the column names in the dataframe
        dd = pd.DataFrame(columns = columns)
        irow = 0
        #run a for loop to get each row that exists
        for row in rows:
            cols = row.find_all('td')
            if (len(cols) >= 28):
                dd.loc[irow] = [
                    cols[0].text.strip(),
                    cols[1].text.strip(),
                    cols[2].text.strip()
                ]
                irow = irow+1
        dd.index += 1
         # Step 2: Sample of CSV being saved. Final Project will have all scraped data loaded in a mysql database
        dd.to_csv(csvfile)
    return dd


# In[ ]:


#Function to Clean the Data to Fit ERD
def cleanStandardBatting(year):    
    csvfile = str(year) + '-standard-batting-cleaned.csv'
    if os.path.isfile(csvfile):
        df = pd.read_csv(csvfile)
    else:
        df = pd.read_csv(str(year) + '-standard-batting.csv')
        #remove unneeded columns
        df = df.drop(columns=['Unnamed: 0', 'Age'])
        ## Rename columns
        df=df.rename(columns={"Name": "batter_name", "Tm": "team"})
        df['year'] = str(year)
        df.index += 1
        df.to_csv(csvfile)
    return df


# In[ ]:


##### Run the Function to Get Data
getStandardBatting(2015)
getStandardBatting(2016)
getStandardBatting(2017)
getStandardBatting(2018)

## Run the Fnciton to Clean Data 
batters_2015 = cleanStandardBatting(2015)
batters_2016 = cleanStandardBatting(2016)
batters_2017 = cleanStandardBatting(2017)
batters_2018 = cleanStandardBatting(2018)


# In[ ]:


## Merge Data 
batter_stats = pd.concat([batters_2015,batters_2016,batters_2017,batters_2018],sort=False)
batter_stats


# In[ ]:


df = batter_stats.drop(columns=['Unnamed: 0'])
df.to_csv('batter_stats.csv')


# Scrape the Pitcher_Stats

# In[ ]:


## Step 1: Creating a function to scrape Starting Pitcher Data
def getStandardPitching(year):
    csvfile = str(year) + '-standard-pitching.csv'
    if os.path.isfile(csvfile):
        dd = pd.read_csv(csvfile)
    else:
        #request the different URLS based on function input
        url = requests.get("https://www.baseball-reference.com/leagues/MLB/"+str(year)+"-standard-pitching.shtml")
        #pull the HTML code
        soup = BeautifulSoup(url.content, "html.parser")
        #Find the table we want to scrap
        div = soup.find('div', id='all_players_standard_pitching')
        #find the contents we want to scrap
        bdiv = bytearray(str(div.contents),'utf-8')
        #pull that contents into a soup object 
        soup2 = BeautifulSoup(bdiv, "html.parser")
        tbody = soup2.find('tbody')
        #find all the rows we want
        rows = tbody.find_all('tr')
        #label all the columns
        columns = ['Name','Age','Tm','Lg','W','L','W-L%','ERA','G','GS','GF','CG','SHO','SV','IP','H',
                   'R','ER','HR','BB','IBB','SO','HBP','BK','WP','BF','ERA+','FIP','WHIP','H9',
                   'BB9','SO9','SO/W']
        #makes these the column names in the dataframe
        dd = pd.DataFrame(columns = columns)
        irow = 0
        #run a for loop to get each row that exists
        for row in rows:
            cols = row.find_all('td')
            if (len(cols) >= 28):
                dd.loc[irow] = [
                    cols[0].text.strip(),
                    cols[1].text.strip(),
                    cols[2].text.strip(),
                    cols[3].text.strip(),
                    cols[4].text.strip(),
                    cols[5].text.strip(),
                    cols[6].text.strip(),
                    cols[7].text.strip(),
                    cols[8].text.strip(),
                    cols[9].text.strip(),
                    cols[10].text.strip(),
                    cols[11].text.strip(),
                    cols[12].text.strip(),
                    cols[13].text.strip(),
                    cols[14].text.strip(),
                    cols[15].text.strip(),
                    cols[16].text.strip(),
                    cols[17].text.strip(),
                    cols[18].text.strip(),
                    cols[19].text.strip(),
                    cols[20].text.strip(),
                    cols[21].text.strip(),
                    cols[22].text.strip(),
                    cols[23].text.strip(),
                    cols[24].text.strip(),
                    cols[25].text.strip(),
                    cols[26].text.strip(),
                    cols[27].text.strip(),
                    cols[28].text.strip(),
                    cols[29].text.strip(),
                    cols[30].text.strip(),
                    cols[31].text.strip(),
                    cols[32].text.strip()
                ]
                irow = irow+1
        dd.index += 1
         # Step 2: Sample of CSV being saved. Final Project will have all scraped data loaded in a mysql database
        dd.to_csv(csvfile)
    return dd


# In[ ]:


#Function to Clean the Data to Fit ERD
def cleanStandardPitching(year):    
    csvfile = str(year) + '-standard-pitching-cleaned.csv'
    if os.path.isfile(csvfile):
        df = pd.read_csv(csvfile)
    else:
        df = pd.read_csv(str(year) + '-standard-pitching.csv')
        #remove unneeded columns
        df = df.drop(columns=['Unnamed: 0','Age', 'Lg','W-L%','CG','BF','ERA+','FIP','WHIP','H9','BB9','SO9','SO/W'])
        ## Rename columns
        df=df.rename(columns={"Name": "pitcher_name", "Tm": "team","W":"wins","L":"loses","ERA":"era","G":"g_played","GS":"g_started","GF":"g_finished","SHO":"shutouts","SV":"saves","IP":"innings_pitched", "H":"hits","R":"runs","ER":"errors","HR":"hr","BB":"bb","IBB":"int_bb","SO":"strike_outs","HBP":"hit_by_pitch","BK":"balks","WP":"wild"})
        ## add the year
        df['year'] = str(year)
        df.index += 1
        df.to_csv(csvfile)
    return df


# In[ ]:


## Run the Function to Get Data
getStandardPitching(2015)
getStandardPitching(2016)
getStandardPitching(2017)
getStandardPitching(2018)

## Run the Fnciton to Clean Data 
pitchers_2015 = cleanStandardPitching(2015)
pitchers_2016 = cleanStandardPitching(2016)
pitchers_2017 = cleanStandardPitching(2017)
pitchers_2018 = cleanStandardPitching(2018)


# In[ ]:


## Merge Data 
pitching_stats = pd.concat([pitchers_2015,pitchers_2016,pitchers_2017,pitchers_2018],sort=False)


# In[ ]:


df = pitching_stats.drop(columns=['Unnamed: 0'])
df.to_csv('pitching_stats.csv')


# Scrape the Pitcher_Info

# In[ ]:


# Define a web scraping function
def get_player_page_urls(url):    
    #request the different URLS based on function input
    url = requests.get(url,headers={'user-agent':'Mozilla/5.0'})
    #pull the HTML code
    soup = BeautifulSoup(url.content, "html.parser")
    #Find the table we want to scrap
    div = soup.find('div', id='all_players_standard_pitching')
    #find the contents we want to scrap
    bdiv = bytearray(str(div.contents),'utf-8')
    #pull that contents into a soup object 
    soup2 = BeautifulSoup(bdiv, "html.parser")
    tbody = soup2.find('tbody')
    #find all the rows we want
    rows = tbody.find_all('tr')
    
    #get all the urls for the players'personal info page 
    player_link=list()
    player_name=list()
    i=0
    for row in rows:
        i+=1
        try:
            str_row=str(row)
            num_row = row.find('th').text
            name = row.find('a').text
            link = re.findall('<a.+?href="(/players/.+?)">',str_row)
            if i % 100 == 0:
                print('saved',i,'page urls','now saving player:',name,"...")
            player_link.append(link)
            player_name.append(name)
        except:
            continue
    return player_link,player_name


# In[ ]:


years = ["2015","2016","2017","2018"]
all_player_link_list = []
for year in years:
    print('Saving year',year,'info...')
    url = "https://www.baseball-reference.com/leagues/MLB/"+year+"-standard-pitching.shtml"
    #begin scraping 
    #call the functions and find all player urls 
    player_link_list,player_name_list = get_player_page_urls(url)
    all_player_link_list.append(player_link_list)
print('Finish saving!')


# In[ ]:


#save the urls into txt files
f = open('all_player_link_list.txt','w')
for link_list in all_player_link_list:
    for i in range(len(link_list)):
        f.write(link_list[i][0])
        f.write('\n')
f.close()


# In[ ]:


# Define a function to get the pitchers' information
def get_player_info(url):
    r=requests.get(url,{'user-agent':'Mozilla/5.0'})
    soup = BeautifulSoup(r.content, 'html.parser')
    #Find the info of items and save
    info_list=[]
    #Find the info of items and save
    
    info_text = soup.find('div',itemtype="https://schema.org/Person").text.replace(u'\n',u' ').replace(u'\xa0',u' ').replace(u'\t',u' ')

    #=========== Find Name =======================================
    try:
        pitcher_name=soup.find('div','players').find('h1').text
    except:
        pitcher_name= 'N/A'


    #=========== Find Height =======================================
    try:

        pitcher_height=soup.find('span',itemprop='height').text.replace('-','.')

    except:
        pitcher_height= 'NULL'


    #=========== Find Weight=======================================
    try:
        pitcher_weight=soup.find('span',itemprop='weight').text.strip('lb')

    except:
        pitcher_weight= 'NULL'

    #=========== Find birthday=======================================
    try:
        birthday=re.findall(r'data-birth=(.*)id',str(soup.find('span',itemprop='birthDate')))[0].replace('"','')
    except:
        birthday= 'NULL'


    #=========== Find Age=======================================
    try:
        now = datetime.now()
        age = str(int(now.year) - int(birthday[:4]))
    except:
        age= 'NULL'    

    #=========== Find Debut Age=======================================
    try:
        debut_age=re.findall(r'Age.?([0-9]{2}.?[0-9]{0,4}d)',info_text)[0]
    except:
        debut_age= 'NULL'

    #=========== Find Birthplace =======================================
    try:
        birthplace =soup.find('span',itemprop='birthPlace').text.strip().replace(u'in',u'').replace(u'\n',u'')
    except:
        birthplace= 'NULL'

    #=========== ***** Find High School=======================================
    try:
        high_school=re.findall(r'High School: ([a-zA-Z_0-9 ]*)',info_text)[0].replace(u'\n',u'').replace(u'\t',u'')
    except:
        high_school= 'NULL'

    #=========== ***** Find School=======================================
    try:
        school=re.findall(r'High School: .* School: ([a-zA-Z_0-9 ]*)',info_text)[0].replace(u'in',u'').replace(u'\n',u'').replace(u'\t',u'')
        if school == high_school:
                school = 'NULL'
    except:
        school= 'NULL'

    #=========== Find Rookie Status=======================================
    try:
        Rookie_Status=re.findall(r'Rookie Status: ([\-a-zA-Z_0-9 .]*)2020 Contract Status',info_text)[0].strip(' ')
    except:
        Rookie_Status= 'NULL'

    # #=========== Find 2020 Contract Status=======================================
    try:
        Contract_Status=re.findall(r'2020 Contract Status: ([\-a-zA-Z_0-9 .]*)Service Time',info_text)[0].strip(' ')
    except:
        Contract_Status= 'NULL'


    # #=========== Find Arb Eligible=======================================
    try:
        Arb_Eligible=re.findall(r'Arb Eligible: ([a-zA-Z_0-9 ]*)',info_text)[0].strip(' ').replace(u'in',u'').replace(u'\n',u'').replace(u'\t',u'')
    except:
        Arb_Eligible= 'NULL'


    # #=========== Find Free Agent=======================================
    try:
        Free_Agent=re.findall(r'Free Agent: ([a-zA-Z_0-9 ]*)Full Name:',info_text)[0].strip(' ').replace(u'in',u'').replace(u'\n',u'').replace(u'\t',u'')
    except:
        Free_Agent= 'NULL'

    # #=========== Find Full Name=======================================
    try:
        Full_Name=re.findall(r'Full Name:</strong>([a-zA-Z_0-9 .]*)',str(soup.find('div',itemtype="https://schema.org/Person")))[0].strip()
    except:
        Full_Name= 'NULL'


    #save all info
    info_list=[pitcher_name,pitcher_height,pitcher_weight,birthday,age,birthplace,
               high_school,school,Rookie_Status,Contract_Status,
               Arb_Eligible,Free_Agent,Full_Name]

    return info_list


# In[ ]:


# Save pitchers' information
all_urls=[]
f = open("all_player_link_list.txt", "r")
for line in f:
    all_urls.append(line.strip('\n'))

all_info=[]
i=0
for link in all_urls:
    i+=1
    if i % 300 == 0:
        print('Saving',i,'player info...')
    url="https://www.baseball-reference.com"+link
    all_info.append(get_player_info(url))


# In[ ]:


#label all the columns
columns = ['Name','Height/ft','Weight/lb','Birthday','Age','Birthplace','High School','School','Rookie Status','2020 Contract Status',
           'Arb Eligible','Free Agent','Full Name']
#makes these the column names in the dataframe
pitcher_info = pd.DataFrame(all_info,columns = columns)
pitcher_info.head(30)


# In[ ]:


pitcher_info.drop_duplicates(subset ="Name",keep = 'first', inplace = True)
pitcher_info.shape


# In[ ]:


pitcher_info.to_csv('pitcher_info.csv')


# The Teams and Pitches table are from Kaggle dataset.

# <center><b><font size="4">Data Cleaning and Table Joining</font></b></center>

# Clean the pitches table

# In[ ]:


# Merge the pitches table with abbats table
pitches=pd.read_csv('pitches_acm.csv')pitches=pd.read_csv('pitches_acm.csv')
ab=pd.read_csv('atbats.csv')
pitches_ab=pitches.merge(right=ab, left_on='ab_id', right_on='ab_id')


# In[ ]:


# Select columns that can be use for analysis
pitches_new=pitches_ab[['pitcher_id','batter_id','g_id','p_throws','inning','o','stand','top','end_speed','spin_rate','b_count','s_count','on_1b','on_2b','on_3b','pitch_num','pitch_type','code']]

# Transform
pitches_new.loc[pitches_new['top'] == True, 'top'] = 'top'
pitches_new.loc[pitches_new['top'] == False, 'top'] = 'bottom'
pitches_new.loc[pitches_new['on_1b'] == True, 'on_1b'] = 1
pitches_new.loc[pitches_new['on_1b'] == False, 'on_1b'] = 0
pitches_new.loc[pitches_new['on_2b'] == True, 'on_2b'] = 1
pitches_new.loc[pitches_new['on_2b'] == False, 'on_2b'] = 0
pitches_new.loc[pitches_new['on_3b'] == True, 'on_3b'] = 1
pitches_new.loc[pitches_new['on_3b'] == False, 'on_3b'] = 0

# Rename column names
pitches_new=pitches_new.rename({'p_throws':'throw','o':'outs','top':'top_bottom','code':'outcome_code'},axis=1)


# In[ ]:


pitches_new.to_csv("/Users/calvin/Desktop/Winter quarter/422/Final Project/mlb-pitch-data-20152018/Pitches.csv",index=False)


# Clean the pitch_stats table

# In[ ]:


# Remove the rows with null value
pitching_stats=pd.read_csv('Pitching_stats_v3.csv')
pitching_stats = pitching_stats.dropna(subset=['pitcher_id'])

pitching_stats.to_csv("/Users/calvin/Desktop/Winter quarter/422/Final Project/mlb-pitch-data-20152018/Pitching_stats.csv",index=False)


# Clean the Batter_Info table

# In[ ]:


# Join the batters' teams with their id and name
batters=pd.read_csv('Batters.csv')
batter_stats=pd.read_csv('batter_stats.csv')
batter_stats.drop(batter_stats.columns[0],axis=1,inplace=True)

team=pd.read_csv('Teams.csv')
team_league=team[['team_abv','league']]
team_id=team[['team_id','team_abv']]

batter_withid=batter_stats.merge(right=batters,left_on='batter_name',right_on='batter_name')
batter_withid=batter_withid.merge(right=team_league,left_on='team',right_on='team_abv')
batter_withid.drop(batter_withid[['team_abv']],axis=1,inplace=True)

batter_withid.to_csv("/Users/calvin/Desktop/Winter quarter/422/Final Project/mlb-pitch-data-20152018/Batter_Info.csv",index=False)


# <center><b><font size="4">Import csv files to MySQL database</font></b></center>

# In[ ]:


# Create Baseball database 
conn=pymysql.connect(host='localhost',user='root', passwd = '') 
cursor = conn.cursor()
db_name = 'Baseball'
cursor.execute('DROP DATABASE IF EXISTS %s' %db_name)
cursor.execute('CREATE DATABASE IF NOT EXISTS %s' %db_name)
cursor.execute('USE Baseball')


# Batter_Info

# In[ ]:


# Create Batter_Info table
conn=pymysql.connect(host='localhost',user='root', passwd = '') 
cursor = conn.cursor()
cursor.execute('USE Baseball')
table_name = 'Batter_Info'
cursor.execute('DROP TABLE IF EXISTS %s' %table_name)
cursor.execute('CREATE TABLE %s(batter_id INT, batter_name varchar(30),team_id tinyint,year int)'%table_name)


# In[ ]:


# Import the batters csv file
conn=pymysql.connect(host='localhost',user='root', passwd = '') 
cursor = conn.cursor()
cursor.execute('USE Baseball')

csv_data = csv.reader(open('Batter_Info.csv'))
# execute and insert the csv into the database.
next(csv_data) # Skip the header
for row in csv_data:
	cursor.execute('INSERT INTO Batter_Info(batter_id,batter_name,team_id,year)''VALUES(%s,%s,%s,%s)',row)
#close the connection to the database.
conn.commit()
cursor.close()
print("CSV has been imported into the database")


# Teams

# In[ ]:


# Create Teams table
conn=pymysql.connect(host='localhost',user='root', passwd = '') 
cursor = conn.cursor()
cursor.execute('USE Baseball')
table_name = 'Teams'
cursor.execute('DROP TABLE IF EXISTS %s' %table_name)
cursor.execute('CREATE TABLE %s(team_id int, team_abv varchar(5), team_name varchar(50),league varchar(20), region varchar(20))'%table_name)


# In[ ]:


# Import the Teams csv file
conn=pymysql.connect(host='localhost',user='root', passwd = '') 
cursor = conn.cursor()
cursor.execute('USE Baseball')

csv_data = csv.reader(open('Teams.csv'))
# execute and insert the csv into the database.
next(csv_data) # Skip the header
for row in csv_data:
	cursor.execute('INSERT INTO Teams(team_id,team_abv,team_name,league,region)''VALUES(%s, %s, %s, %s, %s)',row)
#close the connection to the database.
conn.commit()
cursor.close()
print("CSV has been imported into the database")


# Pitches

# In[ ]:


# Create Pitches table
conn=pymysql.connect(host='localhost',user='root', passwd = '') 
cursor = conn.cursor()
cursor.execute('USE Baseball')
table_name = 'Pitches'
cursor.execute('DROP TABLE IF EXISTS %s' %table_name)
cursor.execute('CREATE TABLE %s(pitcher_id int, batter_id int, g_id int,throw char(1), inning smallint, outs smallint, stand char(1), top_bottom char(10),end_speed float,spin_rate float,b_count tinyint,s_count tinyint,on_1b tinyint,on_2b tinyint,on_3b tinyint,pitch_num tinyint,pitch_type char(3),outcome_code char(2))'%table_name)


# In[ ]:


# Import the Pitches csv file
conn=pymysql.connect(host='localhost',user='root', passwd = '') 
cursor = conn.cursor()
cursor.execute('USE Baseball')

csv_data = csv.reader(open('Pitches.csv'))
# execute and insert the csv into the database.
next(csv_data) # Skip the header
for row in csv_data:
	cursor.execute('INSERT INTO Pitches(pitcher_id,batter_id,g_id,throw,inning,outs,stand,top_bottom,end_speed,spin_rate,b_count,s_count,on_1b,on_2b,on_3b,pitch_num,pitch_type,outcome_code)''VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s)',row)
#close the connection to the database.
conn.commit()
cursor.close()
print("CSV has been imported into the database")


# Note: Due to the size of the dataset, we didn't use the code to import the file. Instead we use the import wizard tool in the MySQLWorkbench.

# In[ ]:


Pitcher_Stats


# In[ ]:


# Create Pitching_Stats table
conn=pymysql.connect(host='localhost',user='root', passwd = '') 
cursor = conn.cursor()
cursor.execute('USE Baseball')
table_name = 'Pitching_Stats'
cursor.execute('DROP TABLE IF EXISTS %s' %table_name)
cursor.execute('CREATE TABLE %s(stats_id int,pitcher_id int,team_id tinyint,wins tinyint,loses tinyint,era float,g_played smallint,g_started smallint,g_finished smallint,shutouts smallint,saves smallint,innings_pitched float,hits smallint,runs smallint,errors smallint,hr tinyint,bb tinyint,int_bb tinyint,strike_outs smallint,hit_by_pitch tinyint,balks tinyint,wild tinyint,year int)'%table_name)


# In[ ]:


# Import the Pitching_Stats csv file
conn=pymysql.connect(host='localhost',user='root', passwd = '') 
cursor = conn.cursor()
cursor.execute('USE Baseball')

csv_data = csv.reader(open('Pitching_Stats.csv'))
# execute and insert the csv into the database.
next(csv_data) # Skip the header
for row in csv_data:
	cursor.execute('INSERT INTO Pitching_Stats(stats_id,pitcher_id,team_id,wins,loses,era,g_played,g_started,g_finished,shutouts,saves,innings_pitched,hits,runs,errors,hr,bb,int_bb,strike_outs,hit_by_pitch,balks,wild,year)''VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s,%s)',row)
#close the connection to the database.
conn.commit()
cursor.close()
print("CSV has been imported into the database")


# Note: Due to the size of the dataset, we didn't use the code to import the file. Instead we use the import wizard tool in the MySQLWorkbench.

# In[ ]:


Pitcher_Info


# In[ ]:


# Create Pitcher_Info table
conn=pymysql.connect(host='localhost',user='root', passwd = '') 
cursor = conn.cursor()
cursor.execute('USE Baseball')
table_name = 'Pitcher_Info'
cursor.execute('DROP TABLE IF EXISTS %s' %table_name)
cursor.execute('CREATE TABLE %s(pitcher_id int,pitcher_name varchar(20),Height/ft float,Weight/lb tinyint,Birthday char(10),Age tinyint,Birthplace varchar(50),High_School varchar(50),School varchar(50),Rookie_Status varchar(50),2020_Contract_Status varchar(20),Arb_Eligible int,Free_Agent int,Full_Name varchar(50))'%table_name)


# In[ ]:


# Import the Pitcher_Info csv file
conn=pymysql.connect(host='localhost',user='root', passwd = '') 
cursor = conn.cursor()
cursor.execute('USE Baseball')

csv_data = csv.reader(open('Pitcher_Info.csv'))
# execute and insert the csv into the database.
next(csv_data) # Skip the header
for row in csv_data:
	cursor.execute('INSERT INTO Pitcher_Info(pitcher_id,pitcher_name,height/ft,weight/lb,birthday,age,school,full_Name)''VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)',row)
#close the connection to the database.
conn.commit()
cursor.close()
print("CSV has been imported into the database")


# Note: Due to the size of the dataset, we didn't use the code to import the file. Instead we use the import wizard tool in the MySQLWorkbench.
