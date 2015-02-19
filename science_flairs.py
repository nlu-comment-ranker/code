#!/usr/bin/env python

import praw
import commentDB
from argparse import ArgumentParser
from sqlalchemy import create_engine, func, distinct
from sqlalchemy.orm import relation, sessionmaker
from requests.exceptions import HTTPError


def process_submission(reddit_id, r, output, fail_list):
    submission_id = reddit_id.split('_')[1]  # cut off 't3_'
    try:
        submission = r.get_submission(submission_id=submission_id)
    except HTTPError:
        fail_list.add(reddit_id)
        return

    flair = submission.link_flair_text
    
    print reddit_id, flair
    output.write('%s,%s\n' % (reddit_id, flair))
    output.flush()


if __name__ == '__main__':
    parser = ArgumentParser(description='Scrape flair of Reddit self-posts')
    parser.add_argument('-u', '--username', type=str,
                        default='nlu_comment_ranker2',
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

    failed = []
    with open(args.output, 'w') as f:
        for s in session.query(commentDB.Submission.sub_id):
            process_submission(s[0], r, f, failed)

        while len(failed) > 0:
            epic_fail = []
            for reddit_id in failed:
                process_submission(reddit_id, r, f, epic_fail)
            failed = epic_fail



