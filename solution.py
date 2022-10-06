import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sentence_transformers import SentenceTransformer, util
import warnings
import pymorphy2
  
warnings.filterwarnings("ignore")

test_jobs = pd.read_csv('for_hack_2022/test/test_jobs.csv', sep=';', names = ["JobId", "Status", "Name", "Region", 'Description',
                                                                                                            'Unk1', 'Unk2', 'Unk3'])
test_candidates = pd.read_csv('for_hack_2022/test/test_candidates.csv', sep=';')
test_edu = pd.read_csv('for_hack_2022/test/test_candidates_education.csv', sep=';')
test_workplaces = pd.read_csv('for_hack_2022/test/test_candidates_workplaces.csv', sep=';')
model = SentenceTransformer('symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli')


def clean(text):

    text=text.lower()
    text=re.sub("</?.*?>"," <> ",text)
    text=re.sub("(\\d|\\W)+"," ",text)

    return text


def citizenship(text):

    with open('countries.txt', 'r') as phil:
        countries = phil.readlines()
    countries = [i.lower().strip() for i in countries]

    t = str(text).lower()
    text = t.split(',')
    if text[0] in countries:
        return text[0]
        
    elif 'рос' in text[0] or 'рф' in text[0] or 'russia' in text[0]:
        return 'россия'
    elif 'узбе' in text[0]:
        return 'узбекистан'
    elif 'кыр' in text[0] or 'кир' in text[0] or 'рк' in text[0] or 'кр' in text[0]:
        return 'киргизия'
    elif 'belarus' in text[0] or 'белоруссия' in text[0] or 'рб' in text[0]:
        return 'беларусь'
    elif 'арм' in text[0]:
        return 'армения'
    elif 'тадж' in text[0]:
        return 'таджикистан'
    elif 'чечня' in text[0] or 'дагестан' in text[0] or 'чнр' in text[0]:
        return 'россия'
    elif 'муж.' in text[0]:
        return 'россия'
    elif 'республика кот-д’ивуар' in text[0]:
        return "кот д'ивуар"

    elif 'молд' in text[0]:
        return 'молдова'
    elif 'внж' in text[0] or 'вид на жительство' in text[0]:
        return 'вид на жительство'

    elif 'казах' in text[0]:
        return 'казахстан'
    elif 'литва' in text[0]:
        return 'литва'

    else: return None


def month_apply(x):
    if 0 < x < 10:  x = '0'+str(x)
    elif x <= 0:  x = '01'
    else: x = str(x)
    return x


def year_apply(x):
    if x <= 0:  x = '1900'
    else:  x = str(x)
    return x


def year_apply_2(x):
    if x <= 0:  x = '2022'
    else:  x = str(x)
    return x    

def cosine_similarity(job, cands, id):

    j = job[job.JobId==id]['Description']
    j = [clean(i) for i in j]
    c = cands['Description']
    c = [clean(i) for i in c] 

    cand_emb = model.encode(c, convert_to_tensor=True)
    job_emb = model.encode(j, convert_to_tensor=True)
    cosine_scores = util.cos_sim(job_emb, cand_emb)
    zipped = zip(cands['CandidateId'],cosine_scores[0].numpy())
    sort = sorted(zipped, key = lambda x: x[1], reverse=True)

    return sort


def preprocess_candidates(candidates, workplaces, edu):

    candidates.DriverLicense.fillna('', inplace=True)
    candidates.DriverLicense = [clean(i) for i in candidates.DriverLicense]
    candidates.DriverLicense = [' '.join(re.findall(r"\b([a-zA-Z]+)\b", i)) for i in candidates.DriverLicense]

    candidates.Langs.fillna('', inplace=True)
    candidates.Langs = [clean(i) for i in candidates.Langs]
    candidates.Langs = [i.replace('базовые знания', '').replace('читаю профессиональную литературу', '').replace('могу проходить интервью', '').replace('родной', '').replace('свободно владею', '')
                        for i in candidates.Langs]

    candidates.Position.fillna('', inplace=True)
    candidates.Position = [clean(i) for i in candidates.Position]
    candidates.drop(['Subway'], axis=1, inplace=True)

    candidates.CandidateRegion.fillna('', inplace=True)
    candidates.CandidateRegion = [clean(i) for i in candidates.CandidateRegion]
    candidates.Citizenship.fillna('', inplace=True)
    candidates.Citizenship = [clean(i) for i in candidates.Citizenship]

    candidates['Citizenship'] = candidates.Citizenship.apply(citizenship)
    candidates.Citizenship.fillna('', inplace=True)

    candidates.Employment.fillna('', inplace=True)
    candidates.Skills.fillna('', inplace=True)
    candidates.Skills = [clean(i) for i in candidates.Skills]
    candidates.Sex.replace({0: "", 1: "женщина", 2: "мужчина"}, inplace=True)
    candidates.Schedule.fillna('', inplace=True)

    workplaces['DateStart'] = pd.to_datetime(workplaces['FromYear'].apply(year_apply) + ' ' + workplaces['FromMonth'].apply(month_apply), format='%Y %m')
    workplaces['DateEnd'] = pd.to_datetime(workplaces['ToYear'].apply(year_apply_2) + ' ' + workplaces['ToMonth'].apply(month_apply), format='%Y %m')

    workplaces.dropna(subset=['Position'], inplace=True) 
    workplaces['Experience'] = (workplaces['DateEnd'] - workplaces['DateStart']).dt.days // 30
    workplaces['Position'] = workplaces['Position'].apply(clean)
    workplaces.drop(labels=['FromYear', 'FromMonth', 'ToYear', 'ToMonth'], axis=1, inplace=True)
    workplaces = workplaces[workplaces.Experience < 800]
    idx = workplaces.groupby(['CandidateId'])['DateEnd'].transform(max)  == workplaces['DateEnd']
    workplaces = workplaces[idx]
    idx = workplaces.groupby(['CandidateId'])['Experience'].transform(max)  == workplaces['Experience']
    workplaces = workplaces[idx]
    workplaces.drop(['DateStart', 'DateEnd', 'Position'], axis=1, inplace=True)

    edu.University.fillna('', inplace=True)
    edu.Faculty.fillna('', inplace=True)
    edu.University = [clean(i) for i in edu.University]
    edu.Faculty = [clean(i) for i in edu.Faculty]
    idx = edu.groupby(['CandidateId'])['GraduateYear'].transform(max)  == edu['GraduateYear']
    edu = edu[idx]
    edu = edu.groupby('CandidateId').first()
    edu.drop(['University', 'GraduateYear'], axis=1, inplace=True)

    candidates = candidates.merge(edu, how='left', on='CandidateId')
    candidates = candidates.merge(workplaces, how='left', on='CandidateId')
    candidates.drop_duplicates(inplace=True)

    text = ['Position', 'Sex', 'Citizenship','Langs', 'DriverLicense', 'Skills', 'Employment', 'Schedule',
            'CandidateRegion', 'Position', 'Faculty']

    df = candidates.copy()
    df['Description'] = df[text].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    df.drop(['Position', 'Sex', 'Citizenship', 'Langs', 'DriverLicense', 'Skills', 'Employment', 'Schedule', 'CandidateRegion', 'Faculty'], axis=1, inplace=True)
    df.drop_duplicates(inplace=True)

    return df


def preprocess_jobs(jobs):

    jobs.Region = [clean(i) for i in jobs.Region]
    jobs.Name = [clean(i) for i in jobs.Name]
    jobs.Description.fillna('', inplace=True)
    jobs.Description = [clean(i) for i in jobs.Description]
    jobs = jobs.loc[jobs.Status < 3]


    def lemmatize(text):

        morph = pymorphy2.MorphAnalyzer()
        words = text.split() 
        res = list()
        for word in words:
            p = morph.parse(word)[0]
            res.append(p.normal_form)

        return res


    def sort_coo(coo_matrix):

        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


    def extract_topn_from_vector(feature_names, sorted_items, topn=10):

        sorted_items = sorted_items[:topn]
        score_vals = []
        feature_vals = []
        for idx, score in sorted_items:        
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])
        results= {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]]=score_vals[idx]
        return results


    def get_stop_words(stop_file_path): 

        with open(stop_file_path, 'r', encoding="utf-8") as f:
            stopwords = f.readlines()
            stop_set = set(m.strip() for m in stopwords)
            return frozenset(stop_set)


    def get_keywords(data,idx):
        tf_idf_vector=tfidf_transformer.transform(cv.transform([data['Description'].iloc[idx]]))
        sorted_items=sort_coo(tf_idf_vector.tocoo())
        keywords=extract_topn_from_vector(feature_names,sorted_items,10)
        return keywords

    docs=jobs['Description'].tolist()
    stopwords = get_stop_words("stopwords-ru.txt")
    cv=CountVectorizer(max_df=0.88,stop_words=stopwords)
    word_count_vector=cv.fit_transform(docs)

    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    feature_names=cv.get_feature_names()

    for i in range(len(jobs)):
        jobs.Description.iloc[i] =  ' '.join(lemmatize(jobs.Description.iloc[i]))
    for i in range(len(jobs)):
        jobs.Description.iloc[i] = " ".join([k for k, v in get_keywords(jobs, i).items() if v >= 0.5])  

    text = ['Name','Region','Description']
    jobs_new = jobs.copy()
    jobs_new['Description'] = jobs_new[text].apply(
        lambda x: ' '.join(x.dropna().astype(str)),
        axis=1
    )
    jobs_new.drop(['Name','Region'], axis=1, inplace=True)
    jobs_new.drop_duplicates(inplace=True)

    return jobs_new


def prediction():
    candidates = preprocess_candidates(test_candidates, test_workplaces, test_edu)
    jobs = preprocess_jobs(test_jobs)
    for i in range(len(jobs)):
        d = pd.DataFrame([list(a) for a in cosine_similarity(jobs, candidates, jobs.JobId.iloc[i])], columns=['CandidateId', 'Score'])
        d.to_csv('result_job_{}.csv'.format(jobs.JobId.iloc[i]), sep=';', header=False, index=False)

if __name__ == '__main__':
    prediction()
