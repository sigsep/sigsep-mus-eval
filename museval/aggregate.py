import pandas
from pathlib import Path
import pandas as pd
import simplejson
import argparse
from urllib.request import urlopen
from jsonschema import validate
import museval
import os
from . version import _version
from decimal import Decimal as D
import numpy as np


class TrackStore(object):
    """
    Holds the metric scores for several frames of one track. 
    
    This is the fundamental building block of other succeeding scores such as `MethodStore` and `EvalStore`. Where as the latter use pandas dataframes, this store is using a simple dict that can easily exported to json using the builtin tools

    Attributes
    ----------
    track_name : str
        name of track.
    win : float, optional
        evaluation window duration in seconds, default to 1 second
    hop : float, optional
        hop length in seconds, defaults to 1 second
    scores : Dict
        Nested Dictionary of all scores
    frames_agg : callable or str
        aggregation function for frames, defaults to `'median' == `np.nanmedian`
    """

    def __init__(self, track_name, win=1, hop=1, frames_agg='median'):
        
        super(TrackStore, self).__init__()
        self.win = win
        self.hop = hop
        if frames_agg == 'median':
            self.frames_agg = np.nanmedian
        elif frames_agg == 'mean':
            self.frames_agg = np.nanmean
        else:
            self.frames_agg = frames_agg
        self.track_name = track_name
        schema_path = os.path.join(
            museval.__path__[0], 'musdb.schema.json'
        )
        with open(schema_path) as json_file:
            self.schema = simplejson.load(json_file)
        self.scores = {
            'targets': [],
            'museval_version': _version
        }

    def add_target(self, target_name, values):
        """add scores of target to the data store

        Parameters
        ----------
        target_name : str
            name of target to be added to list of targets
        values : List(Dict)
            List of framewise data entries, see `musdb.schema.json`
        """
        target_data = {
            'name': target_name,
            'frames': []
        }
        for i, _ in enumerate(values['SDR']):
            frame_data = {
                'time': i * self.hop,
                'duration': self.win,
                'metrics': {
                    "SDR": self._q(values['SDR'][i]),
                    "SIR": self._q(values['SIR'][i]),
                    "SAR": self._q(values['SAR'][i]),
                    "ISR": self._q(values['ISR'][i])
                }
            }
            target_data['frames'].append(frame_data)

        self.scores['targets'].append(target_data)

    @property
    def json(self):
        """formats the track scores as json string

        Returns
        ----------
        json_string : str
            json dump of the scores dictionary
        """
        json_string = simplejson.dumps(
            self.scores,
            indent=2,
            allow_nan=True,
            use_decimal=True
        )
        return json_string

    @property
    def df(self):
        """return track scores as pandas dataframe

        Returns
        ----------
        df : DataFrame
            pandas dataframe object of track scores
        """
        # encode and decode to json first
        return json2df(simplejson.loads(self.json), self.track_name)

    def __repr__(self):
        """Print the frames_aggregated values instead of all frames

        Returns
        ----------
        str
            frames_aggreagted values of all target metrics
        """
        out = ""
        for t in self.scores['targets']:
            out += t['name'].ljust(16) + "==> "
            for metric in ['SDR', 'SIR', 'ISR', 'SAR']:
                out += metric + ":" + \
                    "{:>8.3f}".format(
                        self.frames_agg([np.float(f['metrics'][metric])
                                         for f in t['frames']])
                    ) + "  "
            out += "\n"
        return out

    def validate(self):
        """Validate scores against `musdb.schema`"""
        return validate(self.scores, self.schema)

    def _q(self, number, precision='.00001'):
        """quantiztion of BSSEval values"""
        if np.isinf(number):
            return np.nan
        else:
            return D(D(number).quantize(D(precision)))

    def save(self, path):
        """Saved the track scores as json format"""
        with open(path, 'w+') as f:
            f.write(self.json)


class EvalStore(object):
    """
    Evaluation Storage that holds the scores for multiple tracks.

    This is based on a Pandas DataFrame.

    Attributes
    ----------
    df : DataFrame
        Pandas DataFrame
    frames_agg : str
        aggregation function for frames supports `mean` and `median`, defaults to `median`
    tracks_agg : str
        aggregation function for frames supports `mean` and `median`, defaults to `'median'
    """

    def __init__(self, frames_agg='median', tracks_agg='median'):
        super(EvalStore, self).__init__()
        self.df = pd.DataFrame()
        self.frames_agg = frames_agg
        self.tracks_agg = tracks_agg

    def add_track(self, track):
        """add track score object to dataframe

        Parameters
        ----------
        track : TrackStore or DataFrame
            track store object
        """
        if isinstance(track, TrackStore):
            self.df = self.df.append(track.df, ignore_index=True)
        else:
            self.df = self.df.append(track, ignore_index=True)

    def add_eval_dir(self, path):
        """add precomputed json folder to dataframe

        Parameters
        ----------
        path : str
            path to evaluation results
        """
        p = Path(path)
        if p.exists():
            json_paths = p.glob('test/**/*.json')
            for json_path in json_paths:
                with open(json_path) as json_file:
                    json_string = simplejson.loads(json_file.read())
                track_df = json2df(json_string, json_path.stem)
                self.add_track(track_df)

    def agg_frames_scores(self):
        """aggregates frames scores 

        Returns
        -------
        df_aggregated_frames : GroupBy
             data frame with frames aggregated by mean or median
        """
        df_aggregated_frames_gb = self.df.groupby(
            ['track', 'target', 'metric'])['score']

        if self.frames_agg == 'median':
            df_aggregated_frames = df_aggregated_frames_gb.median()
        elif self.frames_agg == 'mean':
            df_aggregated_frames = df_aggregated_frames_gb.mean()

        return df_aggregated_frames

    def agg_frames_tracks_scores(self):
        """aggregates frames and track scores

        Returns
        -------
        df_aggregated_frames : GroupBy
             data frame with frames and tracks aggregated by mean or median
        """

        df_aggregated_frames = self.agg_frames_scores().reset_index()
        if self.tracks_agg == 'median':
            df_aggregated_tracks = df_aggregated_frames.groupby(
                ['target', 'metric'])['score'].median()
        elif self.tracks_agg == 'mean':
            df_aggregated_tracks = df_aggregated_frames.groupby(
                ['target', 'metric'])['score'].mean()

        return df_aggregated_tracks

    def load(self, path):
        """loads pickled dataframe

        Parameters
        ----------
        path : str
        """
        self.df = pd.read_pickle(path)

    def save(self, path):
        """saves pickled dataframe

        Parameters
        ----------
        path : str
        """
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
                        self.agg_frames_tracks_scores().unstack()[metric][target]) + "  "
            out += "\n"
        return out


class MethodStore(object):
    """
    Holds a pandas DataFrame that stores data for several methods.

    Attributes
    ----------
    df : DataFrame
        Pandas DataFrame
    frames_agg : str
        aggregation function for frames supports `mean` and `median`, defaults to `median`
    tracks_agg : str
        aggregation function for frames supports `mean` and `median`, defaults to `'median'  
    """
    def __init__(self, frames_agg='median', tracks_agg='median'):
        super(MethodStore, self).__init__()
        self.df = pd.DataFrame()
        self.frames_agg = frames_agg
        self.tracks_agg = tracks_agg

    def add_sisec18(self):
        """adds sisec18 participants results to DataFrame.

        Scores will be downloaded on demand.
        """
        print('Downloading SISEC18 Evaluation data...')
        raw_data = urlopen('https://github.com/sigsep/sigsep-mus-2018-analysis/releases/download/v1.0.0/sisec18_mus.pandas')
        print('Done!')
        df_sisec = pd.read_pickle(raw_data, compression=None)
        self.df = self.df.append(df_sisec, ignore_index=True)

    def add_eval_dir(self, path):
        """add precomputed json folder to dataframe.

        The method name will be defined by the basename of provided path

        Parameters
        ----------
        path : str
            path to evaluation results
        """
        method = EvalStore()
        p = Path(path)
        if p.exists():
            json_paths = p.glob('test/**/*.json')
            for json_path in json_paths:
                with open(json_path) as json_file:
                    json_string = simplejson.loads(json_file.read())
                track_df = json2df(json_string, json_path.stem)
                method.add_track(track_df)
        self.add_evalstore(method, p.stem)

    def add_evalstore(self, method, name):
        """add DataFrame

        The method name will be defined by the basename of provided path

        Parameters
        ----------
        method : EvalStore
            EvalStore object
        name : str
            name of method
        """
        df_to_add = method.df
        df_to_add['method'] = name
        self.df = self.df.append(df_to_add, ignore_index=True)
    
    def agg_frames_scores(self):
        """aggregates frames scores

        Returns
        -------
        df_aggregated_frames : GroupBy
             data frame with frames and tracks aggregated by mean or median
        """
        df_aggregated_frames_gb = self.df.groupby(
            ['method', 'track', 'target', 'metric'])['score']

        if self.frames_agg == 'median':
            df_aggregated_frames = df_aggregated_frames_gb.median()
        elif self.frames_agg == 'mean':
            df_aggregated_frames = df_aggregated_frames_gb.mean()

        return df_aggregated_frames

    def agg_frames_tracks_scores(self):
        """aggregates frames and track scores

        Returns
        -------
        df_aggregated_frames : GroupBy
             data frame with frames and tracks aggregated by mean or median
        """
        df_aggregated_frames = self.agg_frames_scores().reset_index()
        if self.tracks_agg == 'median':
            df_aggregated_tracks = df_aggregated_frames.groupby(
                ['method', 'target', 'metric'])['score'].median()
        elif self.tracks_agg == 'mean':
            df_aggregated_tracks = df_aggregated_frames.groupby(
                ['method', 'target', 'metric'])['score'].mean()

        return df_aggregated_tracks

    def load(self, path):
        """loads pickled dataframe

        Parameters
        ----------
        path : str
        """
        self.df = pd.read_pickle(path)

    def save(self, path):
        """saves pickled dataframe

        Parameters
        ----------
        path : str
        """
        self.df.to_pickle(path)


def json2df(json_string, track_name):
    """converts json scores into pandas dataframe

    Parameters
    ----------
    json_string : str
    track_name : str
    """

    df = pd.json_normalize(
        json_string['targets'],
        ['frames'],
        ['name']
    )
    
    df.columns = [col.replace('metrics.', '') for col in df.columns]
    
    df = pd.melt(
        df,
        var_name='metric',
        value_name='score',
        id_vars=['time', 'name'],
        value_vars=['SDR', 'SAR', 'ISR', 'SIR']
    )
    df['track'] = track_name
    df = df.rename(index=str, columns={"name": "target"})
    return df
