import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


valid_row_data_path = 'E:\\数据挖掘\\sna_data\\sna_valid_author_raw.json'
valid_pub_data_path = 'E:\\数据挖掘\\sna_data\\sna_valid_pub.json'

# 合并数据
validate_pub_data = json.load(open(valid_pub_data_path, 'r', encoding='utf-8'))
validate_data = json.load(open(valid_row_data_path, 'r', encoding='utf-8'))
merge_data = {}
for author in validate_data: 
    validate_data[author] = [validate_pub_data[paper_id] for paper_id in validate_data[author]] 

# 验证集数据分析
authors = validate_data.keys()
papers_perauthor = [len(validate_data[author]) for author in validate_data]
print('同名作者数量：', len(authors))
print('涉及的论文数：', np.sum(papers_perauthor))
print('平均论文数量：', np.mean(papers_perauthor))
print('提供的论文数：',len(validate_pub_data))

# 绘制同名作者论文数量
plt.figure(figsize=(20, 8), dpi=80)
x = range(len(authors))

plt.bar(x, papers_perauthor, width=0.8)
plt.xticks(x, authors)
plt.xticks(rotation=270) 
plt.xlabel('测试集同名作者')
plt.ylabel('测试集论文数量（篇）')
for xl, yl in zip(x, papers_perauthor):
    plt.text(xl, yl+0.3, str(yl), ha='center', va='bottom', fontsize=10.5) 
    
plt.gca().hlines(np.mean(papers_perauthor),-1,50,linestyles='--',colors='red',label='平均值')
plt.annotate(u"平均值", xy = (0, np.mean(papers_perauthor)), xytext = (0, 1400),arrowprops=dict(facecolor='red',shrink=0.1,width=2))

plt.show()



# 查看某个同名作者论文情况
author = 'atsushi_takeda'
papers = validate_data[author] 
print('消歧作者名：', author)
print('涉及论文数量：', len(papers))

venue_dict = defaultdict(int)
year_dict = defaultdict(int)
keywords_dict = defaultdict(int)
org_dict = defaultdict(int)

for paper in papers:
    authors = paper['authors']
    venues = paper['venue']
    years = paper['year']
    keywords = paper['keywords']
    
    venue_dict[venues] += 1
    year_dict[years] += 1
    for keyword in keywords:
        keywords_dict[keyword] += 1
    
    for paper_author in authors:
        name = paper_author['name']
        org =  paper_author['org'] if 'org' in paper_author else ""
        org_dict[org] += 1
        
        
# 绘制该名称下论文数据情况
fig = plt.figure(figsize=(20, 20), dpi=80)

ax1 = fig.add_subplot(2,2,1)
x = range(5)
y = [len(papers), len(venue_dict), len(year_dict), len(keywords_dict), len(org_dict)]
s = ['涉及论文数量', '涉及期刊数量', '涉及年份数量', '涉及关键字数量', '涉及机构数量']

plt.bar(x, y, width=0.5)
plt.xticks(x, s, rotation=270)  
plt.xlabel('%s论文数据情况' % author)
plt.ylabel('数量（个）')
for xl, yl in zip(x, y):
    plt.text(xl, yl+0.3, str(yl), ha='center', va='bottom', fontsize=10.5) 

ax2 = fig.add_subplot(2,2,2)
plt.bar(range(len(venue_dict)), venue_dict.values(), width=0.3)
plt.xlabel('%s期刊数据情况' % author)
plt.ylabel('数量（个）')

ax3 = fig.add_subplot(2,2,3)
plt.bar(range(len(year_dict)), year_dict.values(), width=0.5)
plt.xticks(range(len(year_dict)), year_dict.keys(), rotation=270) 
plt.xlabel('%s年份数据情况' % author)
plt.ylabel('数量（个）')

ax4 = fig.add_subplot(2,2,4)
plt.bar(range(len(org_dict)), org_dict.values(), width=0.5)  
plt.xlabel('%s机构数据情况' % author)
plt.ylabel('数量（个）')
plt.show()

# 查看论文作者名中是否包含消歧作者名
print(authors)
for author in validate_data:
    print('disambiguation name: ', author)
    papers = validate_data[author] 
    for paper in papers[:10]:
        print('\npaper id: ' + paper['id'])
        authors = paper['authors']
        for paper_author in authors:
            name = paper_author['name']
            org = paper_author['org']
            print('paper author name: ', name)
            # print(org)
        
    break


import re
# 数据预处理

# 预处理名字
def precessname(name):   
    name = name.lower().replace(' ', '_')
    name = name.replace('.', '_')
    name = name.replace('-', '')
    name = re.sub(r"_{2,}", "_", name) 
   # name = name.replace('.', '_')
    return name

# 预处理机构,简写替换，
def preprocessorg(org):
    if org != "":
        org = org.replace('Sch.', 'School')
        org = org.replace('Dept.', 'Department')
        org = org.replace('Coll.', 'College')
        org = org.replace('Inst.', 'Institute')
        org = org.replace('Univ.', 'University')
        org = org.replace('Lab ', 'Laboratory ')
        org = org.replace('Lab.', 'Laboratory')
        org = org.replace('Natl.', 'National')
        org = org.replace('Comp.', 'Computer')
        org = org.replace('Sci.', 'Science')
        org = org.replace('Tech.', 'Technology')
        org = org.replace('Technol.', 'Technology')
        org = org.replace('Elec.', 'Electronic')
        org = org.replace('Engr.', 'Engineering')
        org = org.replace('Aca.', 'Academy')
        org = org.replace('Syst.', 'Systems')
        org = org.replace('Eng.', 'Engineering')
        org = org.replace('Res.', 'Research')
        org = org.replace('Appl.', 'Applied')
        org = org.replace('Chem.', 'Chemistry')
        org = org.replace('Prep.', 'Petrochemical')
        org = org.replace('Phys.', 'Physics')
        org = org.replace('Phys.', 'Physics')
        org = org.replace('Mech.', 'Mechanics')
        org = org.replace('Mat.', 'Material')
        org = org.replace('Cent.', 'Center')
        org = org.replace('Ctr.', 'Center')
        org = org.replace('Behav.', 'Behavior')
        org = org.replace('Atom.', 'Atomic')
        org = org.split(';')[0]  # 多个机构只取第一个
    return org

#正则去标点
def etl(content):
    content = re.sub("[\s+\.\!\/,;$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", " ", content)
    content = re.sub(r" {2,}", " ", content)
    return content

def get_org(co_authors, author_name):
    for au in co_authors:
        name = precessname(au['name'])
        name = name.split('_')
        if ('_'.join(name) == author_name or '_'.join(name[::-1]) == author_name) and 'org' in au:
            return au['org']
    return ''


# 3. 无监督聚类（根据合作者和机构TFIDF进行相似度聚类） 线上得分：0.2637
import hdbscan
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer

def disambiguate_by_cluster():
    res_dict = {}
    for author in validate_data:
        print(author)
        coauther_orgs = []
        papers = validate_data[author]
        if len(papers) == 0:
            res_dict[author] = []
            continue
        print(len(papers))
        paper_dict = {}
        for paper in papers:
            authors = paper['authors'] 
            names = [precessname(paper_author['name']) for paper_author in authors]
            orgs = [preprocessorg(paper_author['org']) for paper_author in authors if 'org' in paper_author]  
            keywords = [preprocessorg(paper_author['keywords']) for paper_author in authors if 'keywords' in paper_author]
            venue = paper["venue"] if 'venue' in paper else ''
            #print(venue)
            #abstract = paper["abstract"] if 'abstract' in paper else ''
            #coauther_orgs.append(etl(' '.join(names + orgs) + ' '+ abstract))   
            coauther_orgs.append(etl(' '.join(names + orgs) +' ' +' '.join(keywords)+' '+venue)) 
            
        tfidf = TfidfVectorizer().fit_transform(coauther_orgs)
        # sim_mertric = pairwise_distances(tfidf, metric='cosine')
        
        clf = hdbscan.HDBSCAN()  
        s = clf.fit_predict(tfidf)
        #每个样本所属的簇 
        for label, paper in zip(clf.labels_, papers):
            if str(label) not in paper_dict:
                paper_dict[str(label)] = [paper['id']]
            else:
                paper_dict[str(label)].append(paper['id']) 
        res_dict[author] = list(paper_dict.values())
    json.dump(res_dict, open('E:\\数据挖掘\\1\\result.json', 'w', encoding='utf-8'), indent=4)

disambiguate_by_cluster()
