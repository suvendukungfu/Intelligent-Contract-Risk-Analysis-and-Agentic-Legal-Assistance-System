import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface ParsedDocument {
  id: string;
  filename: string;
  text: string;
  page_count: number;
  upload_timestamp: string;
}

export interface Clause {
  id: string;
  document_id: string;
  text: string;
  position: number;
}

export interface RiskPrediction {
  clause_id: string;
  risk_label: 'high_risk' | 'medium_risk' | 'low_risk' | 'no_risk';
  confidence: number;
}

export interface Milestone1Response {
  document_id: string;
  clauses: Array<Clause & RiskPrediction>;
  summary: {
    high_risk: number;
    medium_risk: number;
    low_risk: number;
    no_risk: number;
  };
}

export interface Risk {
  clause_id: string;
  clause_text: string;
  risk_description: string;
  severity: 'high' | 'medium' | 'low';
  explanation: string;
  consequences: string;
  mitigation_actions: string[];
  legal_guidelines: string[];
}

export interface RiskReport {
  contract_summary: string;
  identified_risks: Risk[];
  overall_severity: string;
  legal_disclaimer: string;
}

export interface Milestone2Response {
  document_id: string;
  report: RiskReport;
}

export const analyzeMilestone1 = async (file: File): Promise<Milestone1Response> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post<Milestone1Response>(
    '/api/v1/analyze/milestone1',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return response.data;
};

export const analyzeMilestone2 = async (file: File): Promise<Milestone2Response> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post<Milestone2Response>(
    '/api/v1/analyze/milestone2',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return response.data;
};
