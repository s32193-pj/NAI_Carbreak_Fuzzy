import argparse
import json
import numpy as np
from sklearn.cluster import KMeans


def arg_perser():
    parser = argparse.ArgumentParser(description='Create movie recommendation system')
    parser.add_argument('-m', '--mode', type=str, dest='mode', default='kmeans',)
    parser.add_argument('-k', '--clusters', type=int, dest='clusters', default=3,
                        help='number of user clusters for k-means')
    parser.add_argument('-n', '--count', type=int, dest='count', default=5,
                        help='how many titles to recommend / advise against')

    args = parser.parse_args()
    return args


def load_data():
    with open('ratings_dict.json', encoding='utf-8') as f:
        data = json.load(f)

    return data


def print_hi(data):

    print(data)


def build_rating_matrix(data):
    """Users x titles matrix, NaN where a user has not rated a title."""
    users = list(data.keys())
    titles = sorted({title for ratings in data.values() for title in ratings})
    title_idx = {t: i for i, t in enumerate(titles)}

    matrix = np.full((len(users), len(titles)), np.nan)
    for u, user in enumerate(users):
        for title, rating in data[user].items():
            matrix[u, title_idx[title]] = rating

    return users, titles, matrix


def build_features(matrix):
    """Mean-centre each user's ratings and treat unrated titles as neutral (0).

    Centring removes differences in how generous people are with scores,
    so clustering groups users by taste rather than by rating level.
    Each vector is then scaled to unit length; otherwise users who rated
    many titles sit far from everyone and end up alone in their own cluster.
    """
    user_means = np.nanmean(matrix, axis=1, keepdims=True)
    centred = np.nan_to_num(matrix - user_means, nan=0.0)
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return centred / norms


def score_unseen_titles(matrix, titles, user_row, neighbour_rows):
    """Average rating each unseen title got from the given neighbours."""
    unseen = np.isnan(matrix[user_row])
    scores = []
    for t in np.where(unseen)[0]:
        ratings = matrix[neighbour_rows, t]
        ratings = ratings[~np.isnan(ratings)]
        if len(ratings) > 0:
            scores.append((titles[t], float(ratings.mean()), len(ratings)))
    return scores


def run_kmeans_recommendations(data, user1, n_clusters=3, count=5):

    users, titles, matrix = build_rating_matrix(data)
    features = build_features(matrix)

    n_clusters = max(1, min(n_clusters, len(users)))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(features)

    user_row = users.index(user1)
    user_cluster = labels[user_row]

    # Visit clusters from the user's own one outwards (by distance to centroid),
    # adding their members until there are enough unseen titles to fill both lists.
    distances = np.linalg.norm(kmeans.cluster_centers_ - features[user_row], axis=1)
    cluster_order = [user_cluster] + [c for c in np.argsort(distances) if c != user_cluster]

    neighbour_rows = []
    used_clusters = []
    scores = []
    for c in cluster_order:
        neighbour_rows += [i for i in np.where(labels == c)[0] if i != user_row]
        used_clusters.append(int(c))
        scores = score_unseen_titles(matrix, titles, user_row, neighbour_rows)
        if len(scores) >= 2 * count:
            break

    # Ties broken by how many neighbours rated the title (more votes = more trust).
    best = sorted(scores, key=lambda s: (-s[1], -s[2], s[0]))
    worst = sorted(scores, key=lambda s: (s[1], -s[2], s[0]))
    recommended = best[:count]
    recommended_titles = {s[0] for s in recommended}
    not_recommended = [s for s in worst if s[0] not in recommended_titles][:count]

    return {
        'user': user1,
        'cluster': int(user_cluster),
        'clusters_used': used_clusters,
        'cluster_mates': [users[i] for i in np.where(labels == user_cluster)[0] if i != user_row],
        'neighbours_used': [users[i] for i in neighbour_rows],
        'all_clusters': {int(c): [users[i] for i in np.where(labels == c)[0]]
                         for c in range(n_clusters)},
        'recommended': recommended,
        'not_recommended': not_recommended,
    }


def print_results(results):
    print(f"User: {results['user']}")
    print('\nClusters:')
    for c, members in results['all_clusters'].items():
        marker = '  <- user' if c == results['cluster'] else ''
        print(f"  {c}: {', '.join(members)}{marker}")
    print(f"\nRatings taken from: {', '.join(results['neighbours_used'])}")

    print('\nRecommended:')
    for title, score, votes in results['recommended']:
        print(f'  {title:<45} avg {score:4.1f}  ({votes} rating{"s" if votes > 1 else ""})')

    print('\nNot recommended:')
    for title, score, votes in results['not_recommended']:
        print(f'  {title:<45} avg {score:4.1f}  ({votes} rating{"s" if votes > 1 else ""})')


if __name__ == '__main__':
    arguments = arg_perser()
    program_mode = arguments.mode
    data = load_data()
    user1 = 'Paweł Czapiewski'

    if program_mode == 'kmeans':
        results = run_kmeans_recommendations(data, user1,
                                             n_clusters=arguments.clusters,
                                             count=arguments.count)
        print_results(results)