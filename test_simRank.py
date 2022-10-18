import enum
import unittest
from operator import index
from re import T
import _paths
import glob
import math
import time
import random

import numpy as np
import pandas as pd
import networkx as nx
from neomodel import db
import plotly.express as px
import plotly.graph_objects as go

import oems
import KG_feed_OEM_data as kg
import graph_mining as gm


class TestSimRank(unittest.TestCase):

    def test_simrank(self):
        '''
        check the result from the original paper, general case
        compare result of two functions in nx
        SimRank: A Measure of Structural-Context Similarity 
        '''

        C = nx.DiGraph()
        check = np.array([.414, .331])

        C.add_nodes_from(['Unv', 'ProfA', 'StudA', 'ProfB', 'StudB'])
        C.add_edges_from([
            ('Unv', 'ProfA'), ('Unv', 'ProfB'),
            ('ProfA', 'StudA'),
            ('ProfB', 'StudB'),
            ('StudA', 'Unv'),
            ('StudB', 'ProfB')
        ])

        sim1 = nx.simrank_similarity(
            C, max_iterations=1000, importance_factor=0.8, tolerance=1e-5)
        sim2 = nx.simrank_similarity_numpy(
            C, max_iterations=1000, importance_factor=0.8, tolerance=1e-5)

        s1 = np.array([sim1['ProfA']['ProfB'], sim1['StudA']['StudB']])
        s2 = np.array([sim2[1, 3], sim2[2, 4]])

        c1 = np.allclose(s1, check, rtol=1e-03, atol=1e-03)
        c2 = np.allclose(s2, check, rtol=1e-03, atol=1e-03)

        assert([c1, c2] == [True, True])

    def test_simrank_bipartite(self):
        '''
        check the result from the original paper, bipartite example
        compare result of two functions in nx
        SimRank: A Measure of Structural-Context Similarity 
        '''

        B = nx.Graph()
        check = np.array([0.547, 0.437])

        # Add nodes with the node attribute "bipartite"
        B.add_nodes_from(["A", "B"], bipartite=0)
        B.add_nodes_from(["sugar", "frosting", "egg", "flour"], bipartite=1)

        # Add edges only between nodes of opposite node sets
        B.add_edges_from([
            ("A", "sugar"), ("A", "frosting"), ("A", "egg"),
            ("B", "frosting"), ("B", "egg"), ("B", "flour")])

        top, btw = nx.bipartite.sets(B)

        sim1 = nx.simrank_similarity(
            B, max_iterations=1000, importance_factor=0.8, tolerance=1e-5)
        sim2 = nx.simrank_similarity_numpy(
            B, max_iterations=1000, importance_factor=0.8, tolerance=1e-5)

        s1 = np.array([sim1['A']['B'], sim1['sugar']['flour']])
        s2 = np.array([sim2[0, 1], sim2[2, 5]])

        c1 = np.allclose(s1, check, rtol=1e-03, atol=1e-03)
        c2 = np.allclose(s2, check, rtol=1e-03, atol=1e-03)

        assert([c1, c2] == [True, True])

    def test_simrankpp_simrank(self):
        '''
        check the result from the original paper, 
        check the initial example
        Simrank++: Query rewriting through link analysis of the click graph
        '''
        check = np.array([.619, .437])

        B = nx.Graph()
        B.add_nodes_from([
            "camera", "dcamera",
            "Hp", "Beatsbuy",
            "pc", "tv",
            "flower", "Teleflora", "Orchids"
        ])

        B.add_edges_from([
            ("camera", "Hp"), ("camera", "Beatsbuy"),
            ("dcamera", "Hp"), ("dcamera", "Beatsbuy"),
            ("pc", "Hp"),
            ("tv", "Beatsbuy"),
            ("flower", "Teleflora"), ("flower", "Orchids")
        ])

        sim = nx.simrank_similarity_numpy(
            B, max_iterations=1000, importance_factor=0.8, tolerance=1e-5)

        s1 = np.array([sim[0, 4], sim[4, 5]])
        c1 = np.allclose(s1, check, rtol=1e-03, atol=1e-03)

        assert(c1)

    def test_simrankpp_evidence(self):
        '''
        check the result from the original paper, EVIDENCE, opt=2

        opt=1 , evidence(a, b) = 1 − e^-(|E(a) common E(b)| )
        opt=2, evidence(a, b) = sum(1 /2^i)

        correction for fully connected graph
        Simrank++: Query rewriting through link analysis of the click graph
        '''

        B1 = nx.Graph()
        B2 = nx.Graph()

        B1.add_nodes_from([
            "camera", "dcamera",
            "Hp", "Beatsbuy"])
        B1.add_edges_from([
            ("camera", "Hp"), ("camera", "Beatsbuy"),
            ("dcamera", "Hp"), ("dcamera", "Beatsbuy")])

        B2.add_nodes_from(["camera", "pc", "Hp"])
        B2.add_edges_from([
            ("camera", "Hp"),
            ("pc", "Hp")])

        # -----------------
        # no evidence
        # -----------------
        check0 = np.array([0.6655744, .8])

        sim0_1 = nx.simrank_similarity_numpy(
            B1, max_iterations=7, importance_factor=0.8, tolerance=1e-5)
        sim0_2 = nx.simrank_similarity_numpy(
            B2, max_iterations=7, importance_factor=0.8, tolerance=1e-5)

        c01 = math.isclose(sim0_1[0, 1], check0[0])
        c02 = math.isclose(sim0_2[0, 1], check0[1])
        assert([c01, c02] == [True, True])

        # -----------------
        # opt=2
        # -----------------
        check1 = np.array([.4991808, .4])

        sim2_1 = gm.simrank_pp_similarity_numpy(
            B1, max_iterations=7, importance_factor=0.8, tolerance=1e-5,
            evd_opt=2, sprd_opt=False)
        sim2_2 = gm.simrank_pp_similarity_numpy(
            B2, max_iterations=7, importance_factor=0.8, tolerance=1e-5,
            evd_opt=2, sprd_opt=False)

        c11 = math.isclose(sim2_1[0, 1], check1[0])
        c12 = math.isclose(sim2_2[0, 1], check1[1])

        # print(sim2_1[0, 1], sim2_2[0, 1])
        assert([c11, c12] == [True, True])

        # -----------------
        # opt=1
        # -----------------
        # No result in the paper to compare it with

        sim1_1 = gm.simrank_pp_similarity_numpy(
            B1, max_iterations=7, importance_factor=0.8, tolerance=1e-5,
            evd_opt=1, sprd_opt=False)
        sim1_2 = gm.simrank_pp_similarity_numpy(
            B2, max_iterations=7, importance_factor=0.8, tolerance=1e-5,
            evd_opt=1, sprd_opt=False)

    def test_simrankpp_spread(self):
        '''
        check the result from the original paper, SPREAD

        Higher weight equality is more important than the lower ones

        correction for fully connected graph
        Simrank++: Query rewriting through link analysis of the click graph
        '''

        B1 = nx.Graph()
        B1.add_nodes_from([0, 1, 2, 3])
        B1.add_nodes_from([4])

        eList = [[0, 4], [1, 4], [2, 4], [3, 4]]
        wList = [1000, 1000, 1, 1]

        for i, e in enumerate(eList):
            e0, e1 = e
            B1.add_edge(e0, e1)
            nx.set_edge_attributes(
                B1, {(e0, e1): {'weight': wList[i]}})

        sim1 = gm.simrank_pp_similarity_numpy(
            B1, max_iterations=7, importance_factor=0.8, tolerance=1e-5, sprd_opt='src', evd_opt=2)

        print(sim1[0, 1], sim1[2, 3])
        assert(sim1[0, 1] > sim1[2, 3])

    def test_compare_simrankpp(self):
        '''
        compare the gm.simrank_pp_similarity_numpy with an example from github
        https://github.com/ysong1231/SimRank
        '''
        import sys
        sys.path.insert(0, 'SimRank_ysong')
        from SimRank_ysong.SimRank import SimRank

        B1 = nx.Graph()
        B1.add_nodes_from([0, 1, 2, 3])
        B1.add_nodes_from([4])

        eList = [[0, 4], [1, 4], [2, 4], [3, 4]]
        wList = [1000, 1000, 1, 1]

        for i, e in enumerate(eList):
            e0, e1 = e
            B1.add_edge(e0, e1)
            nx.set_edge_attributes(
                B1, {(e0, e1): {'weight': wList[i]}})

        sim1 = gm.simrank_pp_similarity_numpy(
            B1, max_iterations=100, importance_factor=0.8, tolerance=1e-5, sprd_opt='src', evd_opt=2)

        print('')
        print('sim1 cal:')
        print(sim1[0, 1], sim1[2, 3])
        print(sim1)
        print('')
        print('-----------------------')

        # Get G egdes to dataframe
        edge_df = nx.to_pandas_edgelist(B1)

        # Transform networkx nodes to dataframe
        nodelist = list(B1.nodes(data=True))  # From G to list
        node_df = pd.DataFrame(
            nodelist, columns=['vertex', 'name_attribute'])  # From list to DF
        print(edge_df)
        sr = SimRank.BipartiteSimRank()
        sim21, sim22 = sr.fit(
            edge_df,
            weighted=True,
            C1=.8, C2=.8,
            iterations=100, eps=1e-5,
            node_group1_column='target',
            node_group2_column='source',
            weight_column='weight')
        print('')

        print('')
        print('sim2 cal:')
        print(sim22)
        # print(sim2.loc[0, 1], sim2.loc[2, 3])
        print('')
        print('-----------------------')


if __name__ == '__main__':
    # unittest.main()

    tst = TestSimRank()
    tst.test_compare_simrankpp()
    # tst.test_simrankpp_spread()
    # tst.test_simrankpp_evidence()
