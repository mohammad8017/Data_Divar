import numpy
from numpy.lib.function_base import average
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
from sklearn.decomposition import PCA



def findAvg(brands:list, prices:list, brand):
	count, sum = 0, 0
	for i in range(len(brands)):
		if brands[i] == brand:
			sum += prices[i]
			count += 1

	return sum/count		



def changeNumToEnglish(tmpStr:str): # data cleaning
	return tmpStr.replace('٫','').replace('۰','0').replace('۱','1').replace('۲','2').replace('۳','3').replace('۴','4').replace('۵','5').replace('۶','6').replace('۷','7').replace('۸','8').replace('۹','9').replace(' تومان', '').replace('توافقی', '0').replace('،' , ' ').replace('هستم', '0').replace('قبل از', '')


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
	"last-post-date": 664444352522061
	})
headers = {
'Content-Type': 'application/json',
'Cookie': 'city=tehran; multi-city=tehran%7C'
}

tmpLast = '664444352522061'

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

for step in range(5):

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
usageTmp, yearTmp, brandTmp, priceTmp = [], [], [], []
for ad in allAdds:
	usageTmp.append(int(ad[5]))
	yearTmp.append(1400 - int(ad[3]))
	brandTmp.append(brandExist.index(ad[2]))
	priceTmp.append(int(ad[4]))



plt.boxplot(usageTmp)
plt.title('usage')
plt.show()

q1 = numpy.quantile(usageTmp, 0.25)
q3 = numpy.quantile(usageTmp, 0.75)
med = numpy.median(usageTmp)
iqr = q3-q1
upper_bound = q3+(1.5*iqr)
lower_bound = q1-(1.5*iqr)

tmp = []
for i in range(len(usageTmp)):
	if usageTmp[i] > upper_bound or usageTmp[i] < lower_bound:
		tmp.append(usageTmp.index(usageTmp[i]))
tmp.reverse()		
for i in range(len(tmp)):
	n = usageTmp.pop(tmp[i])
	n = brandTmp.pop(tmp[i])
	n = yearTmp.pop(tmp[i])
	n = priceTmp.pop(tmp[i])		
#--------------------
plt.boxplot(yearTmp)
plt.title('year')
plt.show()

q1 = numpy.quantile(yearTmp, 0.25)
q3 = numpy.quantile(yearTmp, 0.75)
med = numpy.median(yearTmp)
iqr = q3-q1
upper_bound = q3+(1.5*iqr)
lower_bound = q1-(1.5*iqr)

tmp = []
for i in range(len(yearTmp)):
	if yearTmp[i] > upper_bound or yearTmp[i] < lower_bound:
		tmp.append(yearTmp.index(yearTmp[i]))
tmp.reverse()		
for i in range(len(tmp)):
	n = usageTmp.pop(tmp[i])
	n = brandTmp.pop(tmp[i])
	n = yearTmp.pop(tmp[i])
	n = priceTmp.pop(tmp[i])

#--------------------
plt.boxplot(brandTmp)
plt.title('brand')
plt.show()

q1 = numpy.quantile(brandTmp, 0.25)
q3 = numpy.quantile(brandTmp, 0.75)
med = numpy.median(brandTmp)
iqr = q3-q1
upper_bound = q3+(1.5*iqr)
lower_bound = q1-(1.5*iqr)

tmp = []
for i in range(len(brandTmp)):
	if brandTmp[i] > upper_bound or brandTmp[i] < lower_bound:
		tmp.append(brandTmp.index(brandTmp[i]))
tmp.reverse()		
for i in range(len(tmp)):
	n = usageTmp.pop(tmp[i])
	n = brandTmp.pop(tmp[i])
	n = yearTmp.pop(tmp[i])
	n = priceTmp.pop(tmp[i])

#--------------------
print("remove outliers")
print('OK Data :', str(len(priceTmp)))

for i in range(len(yearTmp)):
	if priceTmp[i] > 30000000:
		plt.plot(yearTmp[i], usageTmp[i], 'go')
	elif priceTmp[i] < 2500000:
		plt.plot(yearTmp[i], usageTmp[i], 'ro')
	else:
		plt.plot(yearTmp[i], usageTmp[i], 'yo')		
plt.xlabel('year') 
plt.ylabel('usage')

linear_model=numpy.polyfit(yearTmp, usageTmp,1)
linear_model_fn=numpy.poly1d(linear_model)
x_s=numpy.arange(min(yearTmp), max(yearTmp)+1)
plt.plot(x_s,linear_model_fn(x_s),color="blue")

plt.title(numpy.poly1d(linear_model_fn))
plt.show()


for i in range(len(yearTmp)):
	if priceTmp[i] > 30000000:
		plt.plot(brandTmp[i], usageTmp[i], 'go')
	elif priceTmp[i] < 25000000:
		plt.plot(brandTmp[i], usageTmp[i], 'ro')
	else:
		plt.plot(brandTmp[i], usageTmp[i], 'yo')	
plt.xlabel('brand') 
plt.ylabel('usage')

linear_model=numpy.polyfit(brandTmp, usageTmp,1)
linear_model_fn=numpy.poly1d(linear_model)
x_s=numpy.arange(min(brandTmp), max(brandTmp)+1)
plt.plot(x_s,linear_model_fn(x_s),color="blue")

plt.title(numpy.poly1d(linear_model_fn))
plt.show()


for i in range(len(yearTmp)):
	if priceTmp[i] > 30000000:
		plt.plot(brandTmp[i], yearTmp[i], 'go')
	elif priceTmp[i] < 25000000:
		plt.plot(brandTmp[i], yearTmp[i], 'ro')
	else:
		plt.plot(brandTmp[i], yearTmp[i], 'yo')
plt.xlabel('brand') 
plt.ylabel('year')

linear_model=numpy.polyfit(brandTmp, yearTmp,1)
linear_model_fn=numpy.poly1d(linear_model)
x_s=numpy.arange(min(brandTmp), max(brandTmp)+1)
plt.plot(x_s,linear_model_fn(x_s),color="blue")

plt.title(numpy.poly1d(linear_model_fn))
plt.show()


plot = plt.axes(projection = '3d')
colors = []
for i in range(len(yearTmp)):
	if priceTmp[i] > 3000000:
		colors.append('green')
	elif priceTmp[i] < 2500000:
		colors.append('red')
	else:
		colors.append('yellow')		
colors = numpy.array(colors)	
plot.scatter(yearTmp, usageTmp, brandTmp, c = colors)
plot.set_title('3d plot (year - usage - brand)')
plt.show()


#--------------normalization-----------------

# usageTmp = numpy.asarray(usageTmp)
# usageTmp = preprocessing.normalize([usageTmp])
# usageTmp = (usageTmp.tolist())[0]

# yearTmp = numpy.asarray(yearTmp)
# yearTmp = preprocessing.normalize([yearTmp])
# yearTmp = (yearTmp.tolist())[0]

# brandTmp = numpy.asarray(brandTmp)
# brandTmp = preprocessing.normalize([brandTmp])
# brandTmp = (brandTmp.tolist())[0]

for i in range(len(usageTmp)):
	usageTmp[i] = (usageTmp[i] - min(usageTmp)) / (max(usageTmp) - min(usageTmp))
	yearTmp[i] = (yearTmp[i] - min(yearTmp)) / (max(yearTmp) - min(yearTmp))
	brandTmp[i] = (brandTmp[i] - min(brandTmp)) / (max(brandTmp) - min(brandTmp))

		#---------------------------

for i in range(len(yearTmp)):
	if priceTmp[i] > 30000000:
		plt.plot(yearTmp[i], usageTmp[i], 'go')
	elif priceTmp[i] < 25000000:
		plt.plot(yearTmp[i], usageTmp[i], 'ro')
	else:
		plt.plot(yearTmp[i], usageTmp[i], 'yo')	
plt.xlabel('year') 
plt.ylabel('usage')

linear_model=numpy.polyfit(yearTmp, usageTmp,1)
linear_model_fn=numpy.poly1d(linear_model)
x_s=numpy.arange(min(yearTmp), max(yearTmp)+1)
plt.plot(x_s,linear_model_fn(x_s),color="blue")

plt.title('normalize' + str(numpy.poly1d(linear_model_fn)))
plt.show()

for i in range(len(yearTmp)):
	if priceTmp[i] > 30000000:
		plt.plot(brandTmp[i], usageTmp[i], 'go')
	elif priceTmp[i] < 25000000:
		plt.plot(brandTmp[i], usageTmp[i], 'ro')
	else:
		plt.plot(brandTmp[i], usageTmp[i], 'yo')
plt.xlabel('brand') 
plt.ylabel('usage')

linear_model=numpy.polyfit(brandTmp, usageTmp,1)
linear_model_fn=numpy.poly1d(linear_model)
x_s=numpy.arange(min(brandTmp), max(brandTmp)+1)
plt.plot(x_s,linear_model_fn(x_s),color="blue")

plt.title('normalize' + str(numpy.poly1d(linear_model_fn)))
plt.show()

for i in range(len(yearTmp)):
	if priceTmp[i] > 30000000:
		plt.plot(brandTmp[i], yearTmp[i], 'go')
	elif priceTmp[i] < 25000000:
		plt.plot(brandTmp[i], yearTmp[i], 'ro')
	else:
		plt.plot(brandTmp[i], yearTmp[i], 'yo')
plt.xlabel('brand') 
plt.ylabel('year')

linear_model=numpy.polyfit(brandTmp, yearTmp,1)
linear_model_fn=numpy.poly1d(linear_model)
x_s=numpy.arange(min(brandTmp), max(brandTmp)+1)
plt.plot(x_s,linear_model_fn(x_s),color="blue")

plt.title('normalize' + str(numpy.poly1d(linear_model_fn)))
plt.show()


plot = plt.axes(projection = '3d')
colors = []
for i in range(len(yearTmp)):
	if priceTmp[i] > 3000000:
		colors.append('green')
	elif priceTmp[i] < 2500000:
		colors.append('red')
	else:
		colors.append('yellow')		
colors = numpy.array(colors)		
plot.scatter(yearTmp, usageTmp, brandTmp, c = colors)
plot.set_title('Normalize 3d plot (year - usage - brand)')
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



arr = [usageTmp, yearTmp, brandTmp]
pca = PCA(n_components=3)
p = pca.fit(arr)
x = p.components_

temp = x.tolist()
x1, x2, x3 = x[0], x[1], x[2]


for i in range(len(yearTmp)):
	if priceTmp[i] > average(priceTmp) + 3000000:
		plt.plot(x1[i], x2[i], 'go')
	elif priceTmp[i] < average(priceTmp) - 3000000:
		plt.plot(x1[i], x2[i], 'ro')
	else:
		plt.plot(x1[i], x2[i], 'yo')
plt.title('after PCA')
linear_model=numpy.polyfit(x1, x2,1)
linear_model_fn=numpy.poly1d(linear_model)
x_s=numpy.arange(min(x1), max(x1)+1)
plt.plot(x_s,linear_model_fn(x_s),color="blue")
plt.show()

for i in range(len(yearTmp)):
	if priceTmp[i] > average(priceTmp) + 3000000:
		plt.plot(x1[i], x3[i], 'go')
	elif priceTmp[i] < average(priceTmp) - 3000000:
		plt.plot(x1[i], x3[i], 'ro')
	else:
		plt.plot(x1[i], x3[i], 'yo')
plt.title('after PCA')
linear_model=numpy.polyfit(x1, x3,1)
linear_model_fn=numpy.poly1d(linear_model)
x_s=numpy.arange(min(x1), max(x1)+1)
plt.plot(x_s,linear_model_fn(x_s),color="blue")
plt.show()

for i in range(len(yearTmp)):
	if priceTmp[i] > average(priceTmp) + 3000000:
		plt.plot(x2[i], x3[i], 'go')
	elif priceTmp[i] < average(priceTmp) - 3000000:
		plt.plot(x2[i], x3[i], 'ro')
	else:
		plt.plot(x2[i], x3[i], 'yo')
plt.title('after PCA')
linear_model=numpy.polyfit(x2, x3,1)
linear_model_fn=numpy.poly1d(linear_model)
x_s=numpy.arange(min(x2), max(x2)+1)
plt.plot(x_s,linear_model_fn(x_s),color="blue")
plt.show()


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

