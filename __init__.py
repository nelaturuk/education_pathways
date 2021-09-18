import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, redirect
from wtforms import Form, StringField, SelectField

"""Build the search form, including dropdown menus at the top of the page, from the main datafile."""
class CourseSearchForm(Form):
    df = pd.read_pickle('resources/df_processed.pickle').set_index('Code')
    divisions = [('Any','Any')] + sorted([
        (t,t) for t in set(df.Division.values)
    ])

    departments = [('Any','Any')] + sorted([
        (t,t) for t in set(df.Department.values)
    ])

    campus = [('Any','Any')] + sorted([
        (t,t) for t in set(df.Campus.values)
    ])

    year_choices = [
        (t,t) for t in set(df['Course Level'].values)
    ]
            
    top = [
        ('10','10'),
        ('25','25'),
        ('50','50')
    ]
    select = SelectField('Course Year:', choices=year_choices)
    top = SelectField('',choices=top)
    divisions = SelectField('Division:', choices=divisions)
    departments = SelectField('Department:', choices=departments)
    campuses = SelectField('Campus:', choices=campus)
    search = StringField('Search Terms:')

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    """Homepage is essentially just the course search form. If a post request is received, call the method that finds search results."""
    @app.route('/',methods=['GET','POST'])
    def home():
        search = CourseSearchForm(request.form)
        if request.method == 'POST':
            return search_results(search)
        return render_template('index.html',form=search)

    """Handle the data from the POST request that will go to the main algorithm.
    If we get an empty search, just go back to home.
    Otherwise, pull out the elements of the POST request that are used by the algorithm, and get the results.
    Then, render the results page with a list of pandas tables containing the results for each year.
    Pass the original search to the template as well, so the user can see the context of what they asked for.
    """
    @app.route('/results')
    def search_results(search):
        if search.data['search'] == '' or not search.data['search']:
            return redirect('/')
        results = filter_courses(
            search.data['search'],
            search.data['select'],
            search.data['divisions'],
            search.data['departments'],
            search.data['campuses'],
            search.data['top']
            )

        return render_template('results.html',tables=[t.to_html(classes='data',index=False,na_rep='',render_links=True, escape=False) for t in results],form=search)

    """
    This method shows the information about a single course.
    First, some basic error handling for if a course is passed that does not exist.
    Then, separate the course information into the elements which have specific display functionality and the rest, which we show in a big table.
    Pass all that to render template.
    """
    @app.route('/course/<code>')
    def course(code):

        #If the course code is not present in the dataset, progressively remove the last character until we get a match.
        #For example, if there is no CSC413 then we find the first match that is CSC41.
        #If there are no matches for any character, just go home.
        if code not in df.index:
            while True:
                code = code[:-1]
                if len(code) == 0:
                    return redirect('/')
                t = df[df.index.str.contains(code)]
                if len(t) > 0:
                    code = t.index[0]
                    return redirect('/course/' + code)


        course = df.loc[code]
        #use course network graph to identify pre and post requisites
        pre = G.in_edges(code)
        post = G.out_edges(code)

        excl = course['Exclusion']
        coreq = course['Corequisite']
        aiprereq = course['AIPreReqs']
        majors = course['MajorsOutcomes']
        minors = course['MinorsOutcomes']
        faseavailable = course['FASEAvailable']
        mayberestricted = course['MaybeRestricted']
        terms = course['Term']
        activities = course['Activity']
        course = {k:v for k,v in course.items() if k not in ['Course','Course Level Number','FASEAvailable','MaybeRestricted','URL','Pre-requisites','Exclusion','Corequisite','Recommended Preparation','AIPreReqs','MajorsOutcomes','MinorsOutcomes','Term','Activity'] and v==v}
        return render_template(
            'course.html',
            course=course,
            pre=pre, 
            post=post,
            excl=excl,
            coreq=coreq,
            aip=aiprereq,
            majors=majors,
            minors=minors,
            faseavailable=faseavailable,
            mayberestricted=mayberestricted,
            terms=terms,
            activities=activities,
            zip=zip
            )
    return app

"""
Main algorithm for searching courses. 
In a nutshell: 
1. Split search into phrases e.g. machine learning, biology ==> ['machine learning','biology']
2. Find phrases that occur in the vectorizer (if none, give up). 
3. Find all courses with a nonzero value for each term, and compare all courses against the set of courses with that non-zero value.
4. For every course that is listed as a pre-requisite, add the relevance of its referrer to a list.
5. Take the average of that list and assign it to the pre-requisite.
6. Get the course data for relevant courses in order of score.

"""
def filter_courses(pos_terms, year, division, department, campus, n_return=10):
    print(pos_terms,year)
    n_return=int(n_return) #How many courses are we sending back
    year = int(year) #What year are we primarily looking for
    pos_vals = np.zeros((len(df),))

    #1. Split search into phrases e.g. machine learning, biology ==> ['machine learning','biology']
    #2. Find phrases that occur in the vectorizer (if none, give up). 
    terms = [t for t in pos_terms.split(',') if t.strip() in vectorizer.get_feature_names()]
    print(terms)
    if len(terms) == 0:
        return []
    
    #3. Find all courses with a nonzero value for each term, and compare all courses against the set of courses with that non-zero value.
    #To explain, for each term we look for similarity with all the terms that co-occur with it, to give us a wider scope.
    for term in terms:
        idx = vectorizer.transform([term.strip()]).nonzero()[1][0]
        relevants = course_vectors[:,idx].nonzero()[0]
        pos_vals += np.mean(cosine_similarity(course_vectors,course_vectors[relevants,:]),axis=1)

    #4. For every course that is listed as a pre-requisite, add the relevance of its referrer to a list.
    requisite_vals = defaultdict(list)
    for (k,v),i in zip(df.iterrows(),list(pos_vals)):
        if i>100:
            break
        for col in ['Pre-requisites','Recommended Preparation']:
            for c in v[col]:
                if c in df.index:
                    requisite_vals[c].append(i)

    #5. Take the average of that list and assign it to the pre-requisite.
    for (k,v) in requisite_vals.items():
        requisite_vals[k] = np.mean(v)

    #6. Get the course data for relevant courses in order of score.
    idxs = [t[1] for t in sorted(list(zip(list(pos_vals),list(df.index))),key=lambda x:x[0],reverse=True)]
    tf = df.loc[idxs]

    #7. Separate results by year, starting with the table for the year actually searched for and then decreasing by year. Apply any filters now.
    main_table = tf[tf['Course Level'] == year]
    for name,filter in [('Division',division), ('Department',department), ('Campus',campus)]:
        if filter != 'Any':
            main_table = main_table[main_table[name] == filter]
    tables = [main_table[0:n_return][['Course','Name','Division','Course Description','Department','Course Level']]]
    year -= 1
    while(year > 0):
        tf = df.loc[[t[0] for t in sorted(requisite_vals.items(),key=lambda x: x[1],reverse=True)]]
        tf = tf[tf['Course Level'] == year]
        for name,filter in [('Division',division), ('Department',department), ('Campus',campus)]:
            if filter != 'Any':
                tf = tf[tf[name] == filter]
        tables.append(tf[0:n_return][['Course','Name','Division','Course Description','Department','Course Level']])
        year -= 1
    return tables

with open('resources/course_vectorizer.pickle','rb') as f:
    vectorizer = pickle.load(f)
with open('resources/course_vectors.npz','rb') as f:
    course_vectors = pickle.load(f)
with open('resources/graph.pickle','rb') as f:
    G = nx.read_gpickle(f)
df = pd.read_pickle('resources/df_processed.pickle').set_index('Code')
app = create_app()

if __name__=="__main__":
    app.run(host='0.0.0.0')
