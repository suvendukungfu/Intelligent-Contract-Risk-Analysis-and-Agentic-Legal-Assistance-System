import { useState } from 'react';
import Header from './components/Header';
import MilestoneSelector from './components/MilestoneSelector';
import FileUpload from './components/FileUpload';
import RiskVisualization from './components/RiskVisualization';
import ReportView from './components/ReportView';
import Loading from './components/Loading';
import ErrorMessage from './components/ErrorMessage';
import {
  analyzeMilestone1,
  analyzeMilestone2,
} from './api/client';
import type { Milestone1Response, Milestone2Response } from './api/client';

function App() {
  const [selectedMilestone, setSelectedMilestone] = useState<1 | 2>(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [milestone1Data, setMilestone1Data] = useState<Milestone1Response | null>(null);
  const [milestone2Data, setMilestone2Data] = useState<Milestone2Response | null>(null);

  const handleFileSelect = async (file: File) => {
    setIsLoading(true);
    setError(null);
    setMilestone1Data(null);
    setMilestone2Data(null);

    try {
      if (selectedMilestone === 1) {
        const result = await analyzeMilestone1(file);
        setMilestone1Data(result);
      } else {
        const result = await analyzeMilestone2(file);
        setMilestone2Data(result);
      }
    } catch (err: any) {
      const errorMessage =
        err.response?.data?.error?.message ||
        err.message ||
        'An error occurred while analyzing the contract';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleMilestoneChange = (milestone: 1 | 2) => {
    setSelectedMilestone(milestone);
    setError(null);
    setMilestone1Data(null);
    setMilestone2Data(null);
  };

  return (
    <>
      <Header />
      <div className="container">
        <MilestoneSelector
          selectedMilestone={selectedMilestone}
          onMilestoneChange={handleMilestoneChange}
        />

        <FileUpload onFileSelect={handleFileSelect} isLoading={isLoading} />

        {error && <ErrorMessage message={error} />}

        {isLoading && (
          <Loading
            message={
              selectedMilestone === 1
                ? 'Analyzing contract with ML classifier...'
                : 'Generating comprehensive risk report...'
            }
          />
        )}

        {!isLoading && selectedMilestone === 1 && milestone1Data && (
          <RiskVisualization data={milestone1Data} />
        )}

        {!isLoading && selectedMilestone === 2 && milestone2Data && (
          <ReportView data={milestone2Data} />
        )}
      </div>
    </>
  );
}

export default App;
