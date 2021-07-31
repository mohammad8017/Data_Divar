import numpy
import requests
import json
from datetime import date
import re
import csv
from requests.sessions import extract_cookies_to_jar
import pandas
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import ttest_ind





def changeNumToEnglish(tmpStr:str): # data cleaning
	return tmpStr.replace('٫','').replace('۰','0').replace('۱','1').replace('۲','2').replace('۳','3').replace('۴','4').replace('۵','5').replace('۶','6').replace('۷','7').replace('۸','8').replace('۹','9').replace(' تومان', '').replace('توافقی', '0').replace('،' , ' ').replace('هستم', '0')


def findBrands(txt:str):
	hold = txt.split('enumNames":[')
	holdAll = hold[1].split(']')
	holdAll = holdAll[0].split(',')
	for i in range(len(holdAll)):
		holdAll[i] = holdAll[i].replace('\"', '')

	return holdAll



info = "{ \"phone\":\"9100039730\"}"
headers = {'User-Agent': 'Mozilla/5.0', 'Content-Type': 'application/json', 'x-standard-divar-error': 'true', 'Cookie': 'city=tehran; multi-city=tehran%7C'}
session = requests.Session()


# login = session.post("https://api.divar.ir/v5/auth/authenticate", headers= headers, data=info)
# print(login.text)

# code = input("Enter Code: ")
# tmp = "{ \"phone\":\"9100039730\", \"code\":\""+code+"\"}"


# giveCode = session.post("https://api.divar.ir/v5/auth/confirm", headers=headers, data=tmp)

# token = giveCode.text[10:len(giveCode.text)-2]

# headers['Authorization'] = token
# print('TOKEN:', token)


url = "https://api.divar.ir/v8/search/1/motorcycles"
payload = json.dumps({
	"json_schema": {
		"category": {
		"value": "motorcycles"
		}
	},
	"last-post-date": 663474962109497
	})
headers = {
'Content-Type': 'application/json',
'Cookie': 'city=tehran; multi-city=tehran%7C'
}

tmpLast = '663474962109497'

file = open("C:/Users/Lenovo/Desktop/tamrin/Data_Divar/out.csv", 'w', newline="", encoding="utf-8")
writeFile = csv.writer(file, quoting=csv.QUOTE_NONE)
writeFile.writerow(['title', ' kind of sale', 'brand', 'year', 'price', 'usage', 'explanation'])

tmpToken = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoiMDkxMDAwMzk3MzAiLCJleHAiOjE2Mjg0OTM2ODAuNDE0MjI3LCJ2ZXJpZmllZF90aW1lIjoxNjI3MTk3NjgwLjQxNDIyNSwidXNlci10eXBlIjoicGVyc29uYWwiLCJ1c2VyLXR5cGUtZmEiOiJcdTA2N2VcdTA2NDZcdTA2NDQgXHUwNjM0XHUwNjJlXHUwNjM1XHUwNmNjIn0.T528R8fW1HUErJKRBI-XlaJvpEOuBhdCAEXqIXw7P_o'
headers['Authorization'] = tmpToken

#eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoiMDkxMDAwMzk3MzAiLCJleHAiOjE2Mjg1OTQ4NDEuOTkwMDg1LCJ2ZXJpZmllZF90aW1lIjoxNjI3Mjk4ODQxLjk5MDU0LCJ1c2VyLXR5cGUiOiJwZXJzb25hbCIsInVzZXItdHlwZS1mYSI6Ilx1MDY3ZVx1MDY0Nlx1MDY0NCBcdTA2MzRcdTA2MmVcdTA2MzVcdTA2Y2MifQ.C6EJfXaxDILThoWk7jgunG4NMPe0a_pnXPyfrdn3Qp0

brandExist = []

allLinks = []
counter = 0

allAdds = [] #all adds stored in file

for step in range(1):

	response = requests.request("POST", url, headers=headers, data=payload)
	txt = response.text

	brands = findBrands(txt)
	
	findLinkReg = "\"token\":\"(.*?)\","

	links = re.findall(findLinkReg, txt)
			
	for i in range(len(links)):
		links[i] = 'https://divar.ir/v/' + links[i]

	titleReg = "<title data-react-helmet=\"true\">(.*)<\/title>"
	baseInforeg2 = "<p class=\"kt-unexpandable-row__value\">(.{1,50})<\/p>"
	explanationReg = "<p class=\"kt-description-row__text post-description kt-description-row__text--primary\">([a-zA-Zا-ی\s0-9۰-۹،-]+)"
	phoneNumReg = "<a class=\"kt-unexpandable-row__action kt-text-truncate ltr\" href=\"tel:09397777569\">(۰-۹)+<\/a>"


	#find pattern in data of url and write to csv file
	for link in links:
		if link not in allLinks:
			allLinks.append(link)
		else:
			continue	
		agahi = requests.get(link, headers)
		info = agahi.text

		titleAgahi = re.findall(titleReg, info)
		baseInfo2 = re.findall(baseInforeg2, info)
		exp = re.findall(explanationReg, info)

		try:
			exp[0] = exp[0].replace('\n', ' ')
		except:
			exp = exp    
		allInfo = []
		allInfo.append(changeNumToEnglish(str(titleAgahi[0])))

		for i in range(5):
			if len(baseInfo2) == 4:
				baseInfo2.insert(1,'سایر')
			tmp = baseInfo2[i]
			tmp = list(tmp)
			tmp = ''.join(tmp[:]) 

			# data cleaning
			tmp = changeNumToEnglish(tmp)
			allInfo.append(tmp)


		if baseInfo2[1] not in brandExist:
			brandExist.append(baseInfo2[1])

		tmpExp = ''
		try:
			tmpExp = changeNumToEnglish(exp[0])
		except:
			tmpExp = ' '
		allInfo.append(tmpExp)
		writeFile.writerow(allInfo) 
		allAdds.append(allInfo)
		counter += 1
		print('OK '+ str(step+1) + '.' + str(links.index(link)+1))	

	findLastPostDateReg = "\"last_post_date\":([0-9]+)\,"
	last = re.findall(findLastPostDateReg, txt)
	tmp = list(payload)
	for i in range(15):
		tmp[i+74] = last[0][i]
	tmp = ''.join(tmp)
	payload = tmp	


	tmpLast = last

print("total:", str(counter))
file.close()


brandExist.sort() # all brands we have in data
allAdds.sort(key=lambda x: x[2])

adsDict = {}


# for i in range(len(brandExist)):
# 	tempList = []
# 	for j in range(len(allAdds)):
# 		if allAdds[j][2] == brandExist[i]:
# 			tempList.append(allAdds[j])
# 	adsDict[brandExist[i]] = tempList		


#--------------visualization-----------------
usageTmp, yearTmp, brandTmp = [], [], []
for ad in allAdds:
	usageTmp.append(int(ad[5]))
	yearTmp.append(int(ad[3]))
	brandTmp.append(brandExist.index(ad[2]))


plt.plot(yearTmp, usageTmp, 'ro')
plt.xlabel('year') 
plt.ylabel('usage')
plt.show()

plt.plot(brandTmp, usageTmp, 'ro')
plt.xlabel('brand') 
plt.ylabel('usage')
plt.show()

plt.plot(brandTmp, yearTmp, 'ro')
plt.xlabel('brand') 
plt.ylabel('year')
plt.show()


#--------------normalization-----------------
usageTmp = numpy.asarray(usageTmp)
usageTmp = preprocessing.normalize([usageTmp])
usageTmp = (usageTmp.tolist())[0]

yearTmp = numpy.asarray(yearTmp)
yearTmp = preprocessing.normalize([yearTmp])
yearTmp = (yearTmp.tolist())[0]

brandTmp = numpy.asarray(brandTmp)
brandTmp = preprocessing.normalize([brandTmp])
brandTmp = (brandTmp.tolist())[0]

plt.plot(yearTmp, usageTmp, 'ro')
plt.xlabel('year') 
plt.ylabel('usage')
plt.title('normalize')
plt.show()

plt.plot(brandTmp, usageTmp, 'ro')
plt.xlabel('brand') 
plt.ylabel('usage')
plt.title('normalize')
plt.show()

plt.plot(brandTmp, yearTmp, 'ro')
plt.xlabel('brand') 
plt.ylabel('year')
plt.title('normalize')
plt.show()



T_test = []
t, p = ttest_ind(usageTmp, yearTmp)
T_test.append(p)
t, p = ttest_ind(usageTmp, brandTmp)
T_test.append(p)
t, p = ttest_ind(yearTmp, brandTmp)
T_test.append(p)

ind = T_test.index(max(T_test))
if ind == 0: print('most similarity : usage & year')
elif ind == 1: print('most similarity : usage & brand')
else: print('most similarity : year & brand')
# for i in range(len(allAdds)):
	
# 	usageTmp[i] = (usageTmp[i] - min(usageTmp))/(max(usageTmp) - min(usageTmp))
# 	yearTmp[i] = (yearTmp[i] - min(yearTmp))/(max(yearTmp) - min(yearTmp))
# 	brandTmp[i] = (brandTmp[i] - min(brandTmp))/(max(brandTmp) - min(brandTmp))

# data = pandas.read_csv('C:/Users/Lenovo/Desktop/tamrin/Data_Divar/out.csv')
# X = data[['brand', 'year', 'usage']]
# y = data[['price']]

# regr = linear_model.LinearRegression()
# regr.fit(X, y)

# predictedPrice = regr.predict([['بنلی 300' , 1399, 1000]])

