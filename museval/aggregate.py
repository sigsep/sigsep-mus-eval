import pandas
from pathlib import Path
from pandas.io.json import json_normalize
import pandas as pd
import json
import argparse
from urllib.request import urlopen


class MethodsStore(object):
    def __init__(self, frames_agg='median', tracks_agg='median', tracks=None):
        super(MethodsStore, self).__init__()
        self.df = pd.DataFrame()
        self.frames_agg = frames_agg
        self.tracks_agg = tracks_agg

    def add_sisec18(self):
        print('Downloading SISEC18 Evaluation data...')
        raw_data = urlopen('https://github.com/sigsep/sigsep-mus-2018-analysis/releases/download/v1.0.0/sisec18_mus.pandas')
        print('Done!')
        df_sisec = pd.read_pickle(raw_data, compression=None)
        self.df = self.df.append(df_sisec, ignore_index=True)

    def add_umx(self):
        raw_data = urlopen('.pandas')
        df_sisec = pd.read_pickle(raw_data, compression=None)
        self.df = self.df.append(df_sisec, ignore_index=True)

    def add_method(self, method, method_name='foo'):
        df_to_add = method.df
        df_to_add['method'] = method_name
        self.df = self.df.append(df_to_add, ignore_index=True)
    
    def aggregate_score(self):
        df_aggregated_frames_gb = self.df.groupby(
            ['method', 'track', 'target', 'metric'])

        if self.frames_agg == 'median':
            df_aggregated_frames = df_aggregated_frames_gb.median().reset_index()
        elif self.frames_agg == 'mean':
            df_aggregated_frames = df_aggregated_frames_gb.mean().reset_index()

        if self.tracks_agg == 'median':
            df_aggregated_tracks = df_aggregated_frames.groupby(
                ['method', 'target', 'metric']).median()['score'].unstack()
        elif self.tracks_agg == 'mean':
            df_aggregated_tracks = df_aggregated_frames.groupby(
                ['method', 'target', 'metric']).mean()['score'].unstack()
        
        return df_aggregated_tracks


class EvalStore(object):
    def __init__(self, frames_agg='median', tracks_agg='median', tracks=None):
        super(EvalStore, self).__init__()
        self.df = pd.DataFrame()
        self.frames_agg = frames_agg
        self.tracks_agg = tracks_agg
        if tracks:
            (self.add_track(track) for track in tracks)

    def add_track(self, track):
        self.df = self.df.append(track.df, ignore_index=True)

    def aggregate_score(self, metric, target):
        df_aggregated_frames_gb = self.df.groupby(
            ['track', 'target', 'metric'])

        if self.frames_agg == 'median':
            df_aggregated_frames = df_aggregated_frames_gb.median().reset_index()
        elif self.frames_agg == 'mean':
            df_aggregated_frames = df_aggregated_frames_gb.mean().reset_index()

        if self.tracks_agg == 'median':
            df_aggregated_tracks = df_aggregated_frames.groupby(
                ['target', 'metric']).median()['score'].unstack()
        elif self.tracks_agg == 'mean':
            df_aggregated_tracks = df_aggregated_frames.groupby(
                ['target', 'metric']).mean()['score'].unstack()

        return df_aggregated_tracks[metric][target]

    def load(self, path):
        self.df = pd.read_pickle(path)

    def save(self, path):
        self.df.to_pickle(path)

    def __repr__(self):
        targets = self.df['target'].unique()
        out = "Aggrated Scores ({} over frames, {} over tracks)\n".format(
            self.frames_agg, self.tracks_agg
        )
        for target in targets:
            out += target.ljust(16) + "==> "
            for metric in ['SDR', 'SIR', 'ISR', 'SAR']:
                out += metric + ":" + \
                    "{:>8.3f}".format(
                        self.aggregate_score(metric, target)) + "  "
            out += "\n"
        return out


def json2df(eval_json, track_name):
    json_string = json.loads(eval_json)
    df = json_normalize(
        json_string['targets'],
        ['frames'],
        ['name']
    )
    df = pd.melt(
        pd.concat(
            [
                df.drop(['metrics'], axis=1),
                df['metrics'].apply(pd.Series)
            ],
            axis=1
        ),
        var_name='metric',
        value_name='score',
        id_vars=['time', 'name'],
        value_vars=['SDR', 'SAR', 'ISR', 'SIR']
    )
    df['track'] = track_name
    df = df.rename(index=str, columns={"name": "target"})
    return df
