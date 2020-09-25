import xml.etree.ElementTree
import string
import nltk
import re
import numpy
import time

start_time = time.process_time()

def token_lower_stemming(input_word):
    afterprocess = []
    for word in input_word:

        word_temporary = ""
        for i in range(len(word)):
            if word[i] not in string.punctuation:  # 去除标点符号
                word_temporary += str.lower(word[i])  # 小写化
        wordstemming = nltk.stem.porter.PorterStemmer().stem(word_temporary)  # 词干提取
        if wordstemming not in stopword_list:  # 去除停止词
            afterprocess.append(wordstemming)
    return afterprocess


def invertedindex(inputlist):
    uniqueword, xml_index = [], {}
    for i in range(len(inputlist)):

        print("\r", "Creating inverted index : ", round(i/(len(inputlist)-1)*100,2), "% ", end = "", flush = True)    # 增加进度条
        time.sleep(0.01)

        if i % 2 != 0:  # 因为列表第一元素序号是0，所以除以2不为0的i实际上是偶数个子列表，即文本列表，所以（i-1）子列表就是其对应的文件ID
            for j in range(len(inputlist[i])):
                if inputlist[i][j] not in uniqueword:  # 如果是第一出现的单词
                    uniqueword.append(inputlist[i][j])  # 将新单词并入单词列表
                    xml_index[inputlist[i][j]] = {}  # 定义这个单词为一个字典的键，其值是一个字典
                    xml_index[inputlist[i][j]][inputlist[i - 1][0]] = []  # 将单词的值的键定义为文件ID，其值为列表
                    xml_index[inputlist[i][j]][inputlist[i - 1][0]].append(j + 1)  # 将单词当前的索引附加在列表内。注意要加1，因为列表第一元素序号是0
                else:  # 如果单词不是第一次出现
                    if inputlist[i - 1][0] not in xml_index[inputlist[i][j]]:  # 如果重复的单词是在新的文件ID中出现的
                        xml_index[inputlist[i][j]][inputlist[i - 1][0]] = []  # 为新的文件ID定义一个列表
                        xml_index[inputlist[i][j]][inputlist[i - 1][0]].append(j + 1)  # 将重复单词索引附加在列表上
                    else:  # 如果重复单词出现的不是新的文件ID
                        xml_index[inputlist[i][j]][inputlist[i - 1][0]].append(j + 1)  # 直接将索引附加在列表上
    print("\r", "Creating inverted index : Complete !")    # 因为没有加“end = ""”，所以输出完后就换行。而“\r”只会清除所在行的文本，故本输出会被保留
    return xml_index


def output_invertedindex_as_txt(indext_dict, sorted_key_list):  # 将反转列表输出为txt文本
    outputfile = open('InvertedIndex.txt', 'w')  # 定义一个txt文件，'w'是write
    looptimes_output_inverted_index = 1
    for key in sorted_key_list:

        print("\r", "Output inverted index : ", round(looptimes_output_inverted_index / len(sorted_key_list) * 100, 2), "% ", end="", flush=True)  # 增加进度条
        time.sleep(0.0001)
        looptimes_output_inverted_index += 1

        outputfile.write('%s:\n' % key)  # “\n”是换行，“%s”表示录入字符串，具体内容为后面的“key”
        for docid in indext_dict[key]:
            outputfile.write('\t%s: ' % docid)  # “\t”即为首行缩进，不加“\n”是因为希望索引和文件ID能出现在同一行
            for index_num in range(len(indext_dict[key][docid])):
                if index_num <= (len(indext_dict[key][docid]) - 2):
                    outputfile.write('%d,' % indext_dict[key][docid][index_num])
                if index_num == (len(indext_dict[key][docid]) - 1):
                    outputfile.write('%d\n' % indext_dict[key][docid][index_num])
    outputfile.close()  # 用close命令结束写入，完成输出
    print("\r", "Output inverted index : Complete ! ")


def query_input(inputwholestring):    # 问题整句话以一个字符串的形式输入
    query_separate_letter = ""
    for i in range(len(inputwholestring)):
        query_separate_letter += inputwholestring[i]    # 把问题拆成以各字母为字符转的形式
    return distinguish_input(query_separate_letter)


def distinguish_input(query):    # 以各字母、符号、空格为单独字符串的形式输入
    if query[0] == "#":    # 判断是邻近检索
        result_index = proximity_search(query)    # 以各字母、符号、空格为单独字符串的形式输入
    elif len(query.split()) == 1:    # 判断是单词检索，第一位是否有空格不影响检索
        result_index = []
        normalized_word = token_lower_stemming(query.split())    # token_lower_stemming()输入列表，返回列表
        for key in xml_inverted_index[normalized_word[0]]:
            result_index.append(key)
    elif "AND" in query or "OR" in query:    #判断是布尔检索
        result_index = boolean_search(query.split())    # 以列表形式输入
    else:    #判断是短语检索
        result_index = phrase_rearch(query.split())    # 以列表形式输入
    return result_index


def word_rearch(inputword):    # 以列表形式输入
    return xml_inverted_index[inputword[0]]


def phrase_rearch(inputlist):    # 以列表形式输入
    target_doc = []
    normal_targetword = token_lower_stemming(inputlist)  # 引用预处理函数对文本进行预处理

    word_docid_index = [i for i in range(len(normal_targetword))]
    for i in range(len(normal_targetword)):  # 处理后的短语，分别根据其每个单词，在倒转索引中找到单词对应的索引，统一复制在一个大列表内。单词个数不限
        word_docid_index[i] = xml_inverted_index[normal_targetword[i]]
    for key_01 in word_docid_index[0]:
        for key_02 in word_docid_index[1]:
            if key_01 == key_02:  # 如果两个单词同时出现在一个文件
                for i in word_docid_index[0][key_01]:
                    for j in word_docid_index[1][key_02]:
                        if i - j == -1:  # 在同时出现的文件中，单词索引仅相差为1，及说明找到短语
                            target_doc.append(key_01)  # 将文件名赋值给列表

    print(list(set(sorted(target_doc))))

    print(list)

    return (list(set(sorted(target_doc))))


def proximity_search(inputstring):    # 第一位不能有空格！！以各字母、符号、空格为单独字符串的形式输入
    aimword_keyword = re.findall(r'^#(\d+)\((.+),(.+)\)', inputstring)  # 从输入中提取出距离和关键词

    normal_aimword_keyword = []
    for i in range(1, 3):  # 本循环将短语进行预处理，去除标点、小写化和词干提取
        aimword_stemming = nltk.stem.porter.PorterStemmer().stem(aimword_keyword[0][i].strip())  # 加strip()分词
        if aimword_stemming not in stopword_list:
            normal_aimword_keyword.append(aimword_stemming.lower())

    target_doc_proximity = []
    word_docid_index = ["", ""]  # 定义一个空列表，用来储存关键字的字典索引

    for i in range(2):  # 因为列表第一个元素是数字，即距离范围，非关键词，故从第二个元素开始遍历，而以2+1=3为结束，因为i只会遍历到i=3-1
        word_docid_index[i - 1] = xml_inverted_index[normal_aimword_keyword[i]]

    for key_01 in word_docid_index[0]:
        for key_02 in word_docid_index[1]:
            if key_01 == key_02:  # 如果两个单词同时出现在一个文件
                for i in word_docid_index[0][key_01]:
                    for j in word_docid_index[1][key_02]:
                        if numpy.abs(i - j) <= int(aimword_keyword[0][0]):  # 在同时出现的文件中，单词索引之差在绝对值范围内，即说明找到短语
                            target_doc_proximity.append(key_01)  # 将文件名赋值给列表

    print(target_doc_proximity)
    print(sorted(target_doc_proximity))
    target_doc_proximity01 = target_doc_proximity
    target_doc_proximity01.sort()
    print(target_doc_proximity01)

    return (list(set(target_doc_proximity)))  # 消除重复的文件ID


def boolean_search(inputlist):    # 以列表形式输入
    if "AND" in inputlist:
        if "NOT" in inputlist:    # 布尔词为“AND NOT”
            for boolean_word_rank_andnot in range(len(inputlist)):
                if inputlist[boolean_word_rank_andnot] == "AND":
                    break
            boolean_word_rank_andnot += 1
            boolean_word_01, boolean_word_02 = [], []
            for i in range(boolean_word_rank_andnot - 1):
                boolean_word_01.append(inputlist[i])
            for i in range(boolean_word_rank_andnot + 1, len(inputlist)):
                boolean_word_02.append(inputlist[i])

            if len(boolean_word_01) > 1:
                boolean_word_01_docid = phrase_rearch(boolean_word_01)
            if len(boolean_word_01) == 1:
                boolean_word_01_docid = []
                boolean_word_01_preprocess = token_lower_stemming(boolean_word_01)
                for key in xml_inverted_index[boolean_word_01_preprocess[0]]:
                    boolean_word_01_docid.append(key)

            if len(boolean_word_02) > 1:
                boolean_word_02_docid = phrase_rearch(boolean_word_02)
            if len(boolean_word_02) == 1:
                boolean_word_02_docid = []
                boolean_word_02_preprocess = token_lower_stemming(boolean_word_02)
                for key in xml_inverted_index[boolean_word_02_preprocess[0]]:
                    boolean_word_02_docid.append(key)

            for id_01 in boolean_word_01_docid:
                if id_01 in boolean_word_02_docid:
                    boolean_word_01_docid.remove(id_01)

            return list(set(sorted(boolean_word_01_docid)))

        if "NOT" not in inputlist:  # 布尔词只有“AND”
            for boolean_word_rank_and in range(len(inputlist)):
                if inputlist[boolean_word_rank_and] == "AND":
                    break
            boolean_word_rank_and += 1
            boolean_word_01, boolean_word_02 = [], []
            for i in range(boolean_word_rank_and - 1):
                boolean_word_01.append(inputlist[i])
            for i in range(boolean_word_rank_and, len(inputlist)):
                boolean_word_02.append(inputlist[i])

            if len(boolean_word_01) > 1:
                boolean_word_01_docid = phrase_rearch(boolean_word_01)
            if len(boolean_word_01) == 1:
                boolean_word_01_docid = []
                boolean_word_01_preprocess = token_lower_stemming(boolean_word_01)
                for key in xml_inverted_index[boolean_word_01_preprocess[0]]:
                    boolean_word_01_docid.append(key)

            if len(boolean_word_02) > 1:
                boolean_word_02_docid = phrase_rearch(boolean_word_02)
            if len(boolean_word_02) == 1:
                boolean_word_02_docid = []
                boolean_word_02_preprocess = token_lower_stemming(boolean_word_02)
                for key in xml_inverted_index[boolean_word_02_preprocess[0]]:
                    boolean_word_02_docid.append(key)

            boolean_and_docid = []

            for id_01 in boolean_word_01_docid:
                if id_01 in boolean_word_02_docid:
                    boolean_and_docid.append(id_01)
            return list(set(sorted(boolean_and_docid)))

    if "OR" in inputlist:
        for boolean_word_rank_or in range(len(inputlist)):
            if inputlist[boolean_word_rank_or] == "OR":
                break
        boolean_word_rank_or += 1
        boolean_word_01, boolean_word_02 = [], []
        for i in range(boolean_word_rank_or - 1):
            boolean_word_01.append(inputlist[i])
        for i in range(boolean_word_rank_or, len(inputlist)):
            boolean_word_02.append(inputlist[i])

        if len(boolean_word_01) > 1:
            boolean_word_01_docid = phrase_rearch(boolean_word_01)
        if len(boolean_word_01) == 1:
            boolean_word_01_docid = []
            boolean_word_01_preprocess = token_lower_stemming(boolean_word_01)
            for key in xml_inverted_index[boolean_word_01_preprocess[0]]:
                boolean_word_01_docid.append(key)

        if len(boolean_word_02) > 1:
            boolean_word_02_docid = phrase_rearch(boolean_word_02)
        if len(boolean_word_02) == 1:
            boolean_word_02_docid = []
            boolean_word_02_preprocess = token_lower_stemming(boolean_word_02)
            for key in xml_inverted_index[boolean_word_02_preprocess[0]]:
                boolean_word_02_docid.append(key)

        boolean_or_docid = boolean_word_01_docid + boolean_word_02_docid
        return list(set(sorted(boolean_or_docid)))


def output_boolean_query_result(query_result):    # 以二维列表输入
    outputfile = open('BooleanSearch.txt', 'w')
    for list_num in range(len(query_result)):

        print("\r", "Output boolean search : ", round(list_num / len(query_result) * 100, 2), "% ", end="", flush = True)  # 增加进度条
        time.sleep(0.0001)

        if list_num % 2 != 0:
            for docid_num in range(len(query_result[list_num])):
                outputfile.write('%s' % query_result[list_num - 1][0])
                outputfile.write('\t0')
                outputfile.write('\t%s' % query_result[list_num][docid_num])
                outputfile.write('\t0')
                outputfile.write('\t1')
                outputfile.write('\t0\n')
    outputfile.close()
    print("\r", "Output boolean search result : Complete !")    # 因为没有加“end = ""”，所以输出完后就换行。而“\r”只会清除所在行的文本，故本输出会被保留


# 导入XML文件
#tree = xml.etree.ElementTree.parse("C:/Users/ps/Desktop/TTDS/Metrial/lab02/trec.sample.xml")    # 练习资源
tree = xml.etree.ElementTree.parse("trec.5000.xml")    # 作业资源
root = tree.getroot()

stopword_list, xml_text = [], []

stopword = open("Stopword.txt", 'r')
for word in stopword:
    stopword_list.append(word.strip('\n'))

looptimes_import_xml = 1

for doc in root.findall("DOC"):

    print("\r", "Importing documents : No.", looptimes_import_xml, end="", flush = True)  # 增加进度条
    time.sleep(0.001)
    looptimes_import_xml += 1

    docno = doc.find("DOCNO").text
    headline = doc.find("HEADLINE").text
    text = doc.find("TEXT").text

    xml_docno = re.findall(r'\b[A-Za-z0-9]+\b', docno.strip())  # strip()是去除空格，如果用split()的话，万一两个单词间有多个空格，其只能去除一个，就会有很多空元素产生
    xml_text_word = re.findall(r'\b[A-Za-z0-9]+\b', text.strip())  # r'\b[A-Za-z0-9]+\b'是将每个单词左右两边的标点去除
    xml_headline_word, xml_text_word = re.findall(r'\b[A-Za-z0-9]+\b', headline.strip()), re.findall(r'\b[A-Za-z0-9]+\b', text.strip())
    xml_word = xml_headline_word + xml_text_word  # 用列表的加法，合并两个文本

    xml_text.append(xml_docno)  # 将第doc个文件ID以列表形式赋值给列表
    xml_text.append(token_lower_stemming(xml_word))  # 将第doc个，经过预处理的文本以列表形式赋值给列表，这样就能产生二维列表，第i是文件ID，i+1是ID对应的文本
print('\n', "Importing documents : Complete ! ")    # 因为没有加“end = ""”，所以输出完后就换行。而“\r”只会清除所在行的文本，故本行输出会被保留


# 反转索引
xml_inverted_index = invertedindex(xml_text)    # 用函数生成反转索引
xml_inverted_index_sorted_key = sorted(xml_inverted_index.keys(), reverse = False)    # 将倒转索引中每个字都以列表形式输出来

output_invertedindex = output_invertedindex_as_txt(xml_inverted_index, xml_inverted_index_sorted_key)    # 引用函数，将反转索引以txt形式导出


# 布尔检索
search_query_open = open('queries.boolean.txt','r')    # 打开记有问题的txt文件
readlines = search_query_open.readlines()    # 如果是.readline()，即没有“s”，则只会扫描txt文件第一行文本
lines = []
query_result = []
for line in readlines:
    search_query = re.findall(r'^(\d+) (.+)', line)    # (\d+)是问题编号，(.+)是问题，包括标点和空格
    query_result.append([search_query[0][0]])    #附加问题序号
    query_result.append(query_input(search_query[0][1]))    # 附加问题回答的搜索结果
print("\r", "Boolean search : Complete !")    # 因为没有加“end = ""”，所以输出完后就换行。而“\r”只会清除所在行的文本，故本输出会被保留

output_boolean_query_result(query_result)    # 用自定义函数导出布尔检索结果

#
# # 排序
# search_query_open = open('queries.boolean_00.txt','r')    # 打开记有问题的txt文件
# readlines = search_query_open.readlines()    # 如果是.readline()，即没有“s”，则只会扫描txt文件第一行文本
# lines = []
# query_result = []
# for line in readlines:
#     search_query = re.findall(r'^(\d+) (.+)', line)    # (\d+)是问题编号，(.+)是问题，包括标点和空格
#     query_result.append([search_query[0][0]])    #附加问题序号
#     query_result.append(query_input(search_query[0][1]))    # 附加问题回答的搜索结果
# print("\r", "Boolean search : Complete !")    # 因为没有加“end = ""”，所以输出完后就换行。而“\r”只会清除所在行的文本，故本输出会被保留



end_time = time.process_time()
print("Process time : ", end_time - start_time, "s")


