import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

class exploratory_data_analysis:
    def __init__(self) -> None:
        self.data = pd.read_csv('Data/PET_PRI_GND_DCUS_NUS_W.csv')

    def eda(self):
        logfile = PdfPages('MlProjectEDA.pdf')

        for col in self.data.columns[1:13]:
            fig = plt.figure(figsize=(14, 6))
            plt.title(f'Distplot for {col}')
            sns.distplot(self.data[col])
            logfile.savefig(fig)
            plt.show()

        top = self.data
        top.loc[-1] = top.max(axis = 0)
        top = top.tail(1)
        top = top.drop('Date' , axis = 1)

        # comparison of the most expensive gas prices
        fig = plt.figure(figsize=(14, 8))
        sns.set_style("darkgrid")

        plt.barh('A1', top['A1'], color='blue')
        plt.barh('A2', top['A2'], color='blue')
        plt.barh('A3', top['A3'], color='blue')
        plt.barh('R1', top['R1'], color='green')
        plt.barh('R2', top['R2'], color='green')
        plt.barh('R3', top['R3'], color='green')
        plt.barh('M1', top['M1'], color='orange')
        plt.barh('M2', top['M2'], color='orange')
        plt.barh('M3', top['M3'], color='orange')
        plt.barh('P1', top['P1'], color='red')
        plt.barh('P2', top['P2'], color='red')
        plt.barh('P3', top['P3'], color='red')

        plt.xticks(np.arange(0, 5, step=0.5))
        plt.title("Comparison of most expensive gas prices", fontsize=20, fontweight='bold')
        plt.ylabel('Type', fontsize=14, fontweight='bold')
        plt.xlabel("Price", fontsize=14, fontweight='bold')
        logfile.savefig(fig)
        plt.show()

        # Average of all formulations retails gasoline prices
        data = self.data
        data['year'] = pd.DatetimeIndex(data['Date']).year
        data.drop(-1 , inplace = True)

        annual_avg = data.groupby('year').mean()
        annual_avg = annual_avg.reset_index()

        fig = plt.figure(figsize=(14, 8))
        sns.set_style("darkgrid")
        plt.plot(annual_avg['year'], annual_avg['A1'])
        plt.plot(annual_avg['year'], annual_avg['R1'])
        plt.plot(annual_avg['year'], annual_avg['M1'])
        plt.plot(annual_avg['year'], annual_avg['P1'])

        plt.xticks(np.arange(1995, 2021, step=3))
        plt.legend(['A1', 'R1', 'M1', 'P1'], loc='upper right')
        plt.title("Average pf All Formulations Retail Gasoline Prices", fontsize=18, fontweight='bold')
        plt.xlabel("Year", fontsize=14, fontweight='bold')
        plt.ylabel("Price ($/gallon)", fontsize=14, fontweight='bold')
        logfile.savefig(fig)
        plt.show()

        # Average of conventional Retail Gasoline Prices

        fig = plt.figure(figsize=(14, 8))
        sns.set_style("darkgrid")
        plt.plot(annual_avg['year'], annual_avg['A2'])
        plt.plot(annual_avg['year'], annual_avg['R2'])
        plt.plot(annual_avg['year'], annual_avg['M2'])
        plt.plot(annual_avg['year'], annual_avg['P2'])

        plt.xticks(np.arange(1995, 2021, step=3))
        plt.legend(['A1', 'R1', 'M1', 'P1'], loc='upper right')
        plt.title("Average pf All Formulations Retail Gasoline Prices", fontsize=18, fontweight='bold')
        plt.xlabel("Year", fontsize=14, fontweight='bold')
        plt.ylabel("Price ($/gallon)", fontsize=14, fontweight='bold')
        logfile.savefig(fig)
        plt.show()

        # Average of all Reformulated Retail Gasoline Prices

        fig = plt.figure(figsize=(14, 8))
        sns.set_style("darkgrid")
        plt.plot(annual_avg['year'], annual_avg['A3'])
        plt.plot(annual_avg['year'], annual_avg['R3'])
        plt.plot(annual_avg['year'], annual_avg['M3'])
        plt.plot(annual_avg['year'], annual_avg['P3'])

        plt.xticks(np.arange(1995, 2021, step=3))
        plt.legend(['A1', 'R1', 'M1', 'P1'], loc='upper right')
        plt.title("Average pf All Formulations Retail Gasoline Prices", fontsize=18, fontweight='bold')
        plt.xlabel("Year", fontsize=14, fontweight='bold')
        plt.ylabel("Price ($/gallon)", fontsize=14, fontweight='bold')
        logfile.savefig(fig)
        plt.show()
        logfile.close()



