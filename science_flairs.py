#!/usr/bin/env python

import praw
import commentDB
from argparse import ArgumentParser
from sqlalchemy import create_engine, func, distinct
from sqlalchemy.orm import relation, sessionmaker
from requests.exceptions import HTTPError

if __name__ == '__main__':
    parser = ArgumentParser(description='Scrape flair of Reddit self-posts')
    parser.add_argument('-u', '--username', type=str,
                        default='nlu_comment_ranker',
                        help='reddit username')
    parser.add_argument('-p', '--password', type=str,
                        default='cardinal_cs224u',
                        help='reddit password')
    parser.add_argument('-d', '--dbfile', type=str,
                        default='redditDB-jun6.db',
                        help="SQLite database file to read")
    parser.add_argument('-o', '--output', type=str,
                        default='flairs.csv',
                        help='File to write flairs to')

    args = parser.parse_args()

    user_agent = ("NLU project: flair scraper "
                  "by /u/nlu_comment_ranker (smnguyen@stanford.edu)")
    r = praw.Reddit(user_agent=user_agent)
    r.login(username=args.username, password=args.password)

    engine = create_engine('sqlite:///' + args.dbfile, echo=True)
    commentDB.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    flair_map = {}
    for s in session.query(commentDB.Submission.sub_id):
        reddit_id = s[0]
        submission_id = reddit_id.split('_')[1]  # cut off 't3_'
        submission = r.get_submission(submission_id=submission_id)
        flair = submission.link_flair_text
        print reddit_id, flair
        flair_map[reddit_id] = flair

    with open(args.output, 'w') as f:
        for sub_id in flair_map:
            f.write('%s,%s\n' % (sub_id, flair_map[sub_id]))

