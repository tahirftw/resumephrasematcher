# import libraries
import os
from os import listdir
from os.path import isfile, join
from io import StringIO
import pandas as pd
from collections import Counter
import fitz
import spacy

nlp = spacy.load("en_core_web_sm")
from spacy.matcher import PhraseMatcher
import matplotlib.pyplot as plt

# function reads the resumes from the specified folder
# the text on each page of the resume is read and appended in the array
mypath = r'path_to_resume_folder'  # enter your path here where you saved the resumes
pdfdocs = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]


def pdfextract2(resume):
    doc = fitz.open(resume)
    text = []
    for page in doc:
        t = page.get_text()
        text.append(t)
    return text


# function that performs phrase matching and builds the employee skill profile
def create_profile(resume):
    text = pdfextract2(resume)
    text = str(text)
    text = text.replace("\\n", "")
    text = text.lower()

    # read the categorized keywords/skills stored in the csv file
    keyword_dict = pd.read_csv(r'path_to_skills_csv', encoding='ansi')

    # match the skills found in the resume texts with the skills in the csv file
    stats_words = [nlp(text) for text in keyword_dict['Statistics'].dropna(axis=0)]
    math_words = [nlp(text) for text in keyword_dict['Mathematics'].dropna(axis=0)]
    ai_words = [nlp(text) for text in keyword_dict['Artificial Intelligence'].dropna(axis=0)]
    programming_words = [nlp(text) for text in keyword_dict['Programming'].dropna(axis=0)]
    cloud_computing_words = [nlp(text) for text in keyword_dict['Cloud Computing'].dropna(axis=0)]

    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('Stats', None, *stats_words)
    matcher.add('Math', None, *math_words)
    matcher.add('AI', None, *ai_words)
    matcher.add('Prog', None, *programming_words)
    matcher.add('CloudComp', None, *cloud_computing_words)
    doc = nlp(text)

    d = []
    matches = matcher(doc)
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]
        span = doc[start:end]
        d.append((rule_id, span.text))
    keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i, j in Counter(d).items())

    # convert string of keywords/skills to dataframe
    df = pd.read_csv(StringIO(keywords), names=['Skills_List'])
    df1 = pd.DataFrame(df.Skills_List.str.split(' ', 1).tolist(), columns=['Category', 'Keyword'])
    df2 = pd.DataFrame(df1.Keyword.str.split('(', 1).tolist(), columns=['Keyword', 'Count'])
    df3 = pd.concat([df1['Category'], df2['Keyword'], df2['Count']], axis=1)
    df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))

    base = os.path.basename(resume)
    filename = os.path.splitext(base)[0]

    # build the employee profile
    name = filename.split('_')
    name2 = name[0]
    name2 = name2.lower()

    # converting str to dataframe
    name3 = pd.read_csv(StringIO(name2), names=['Employee Name'])

    dataf = pd.concat([name3['Employee Name'], df3['Category'], df3['Keyword'], df3['Count']], axis=1)
    dataf['Employee Name'].fillna(dataf['Employee Name'].iloc[0], inplace=True)

    return dataf


# code to execute/call the above functions
text_content = pd.DataFrame()
for resume in pdfdocs:
    dat = create_profile(resume)
    text_content = text_content.append(dat)

# count words under each category and visualize it through Matplotlib
text_content2 = text_content['Keyword'].groupby(
    [text_content['Employee Name'], text_content['Category']]).count().unstack()
text_content2.reset_index(inplace=True)
text_content2.fillna(0, inplace=True)
new_data = text_content2.iloc[:, 1:]
new_data.index = text_content2['Employee Name']

plt.rcParams.update({'font.size': 10})
ax = new_data.plot.barh(title="Skills per category", legend=False, figsize=(25, 7), stacked=True)
labels = []
for j in new_data.columns:
    for i in new_data.index:
        label = str(j) + ": " + str(new_data.loc[i][j])
        labels.append(label)
patches = ax.patches
for label, rect in zip(labels, patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x()
        y = rect.get_y()
        height = rect.get_height()
        ax.text(x + width / 2., y + height / 2., label, ha='center', va='center')
plt.show()

# Determine the best fit for the new position based on required skills
required_skills = [
    'data analysis', 'ai & ml', 'cloud computing', 'cybersecurity', 'programming', 'digital marketing',
    'it infrastructure', 'enterprise architecture', 'project management', 'change management',
    'strategic planning', 'budget management', 'vendor management', 'agile methodologies', 'leadership',
    'communication', 'problem solving', 'critical thinking', 'team collaboration', 'customer focus',
    'financial services', 'regulatory compliance', 'digital banking trends', 'fintech innovations'
]

fit_scores = {}
for employee in text_content['Employee Name'].unique():
    employee_skills = text_content[text_content['Employee Name'] == employee]['Keyword'].str.lower().tolist()
    fit_score = len(set(employee_skills).intersection(set(required_skills)))
    fit_scores[employee] = fit_score

best_fit_employee = max(fit_scores, key=fit_scores.get)

# Summary of findings
print(f"Best fit employee for the Digital Transformation Manager position: {best_fit_employee}")
print(f"Skills match: {fit_scores[best_fit_employee]} out of {len(required_skills)} required skills")

# Display the plot
plt.show()
