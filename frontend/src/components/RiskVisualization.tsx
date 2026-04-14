import React, { useState } from 'react';
import type { Milestone1Response } from '../api/client';

interface RiskVisualizationProps {
  data: Milestone1Response;
}

const RiskVisualization: React.FC<RiskVisualizationProps> = ({ data }) => {
  const [filter, setFilter] = useState<string | null>(null);

  const filteredClauses = filter
    ? data.clauses.filter((clause) => clause.risk_label === filter)
    : data.clauses;

  const getRiskClass = (riskLabel: string) => {
    switch (riskLabel) {
      case 'high_risk':
        return 'high-risk';
      case 'medium_risk':
        return 'medium-risk';
      case 'low_risk':
        return 'low-risk';
      case 'no_risk':
        return 'no-risk';
      default:
        return '';
    }
  };

  const getRiskLabel = (riskLabel: string) => {
    switch (riskLabel) {
      case 'high_risk':
        return 'High Risk';
      case 'medium_risk':
        return 'Medium Risk';
      case 'low_risk':
        return 'Low Risk';
      case 'no_risk':
        return 'No Risk';
      default:
        return riskLabel;
    }
  };

  return (
    <div className="risk-visualization">
      <h2>Risk Analysis Results</h2>

      {/* Summary Cards */}
      <div className="risk-summary">
        <div className="risk-summary-card high">
          <div className="risk-summary-count">{data.summary.high_risk}</div>
          <div className="risk-summary-label">High Risk</div>
        </div>
        <div className="risk-summary-card medium">
          <div className="risk-summary-count">{data.summary.medium_risk}</div>
          <div className="risk-summary-label">Medium Risk</div>
        </div>
        <div className="risk-summary-card low">
          <div className="risk-summary-count">{data.summary.low_risk}</div>
          <div className="risk-summary-label">Low Risk</div>
        </div>
        <div className="risk-summary-card none">
          <div className="risk-summary-count">{data.summary.no_risk}</div>
          <div className="risk-summary-label">No Risk</div>
        </div>
      </div>

      {/* Filters */}
      <div className="risk-filters">
        <button
          className={`filter-button ${filter === null ? 'active' : ''}`}
          onClick={() => setFilter(null)}
        >
          All Clauses
        </button>
        <button
          className={`filter-button ${filter === 'high_risk' ? 'active' : ''}`}
          onClick={() => setFilter('high_risk')}
        >
          High Risk
        </button>
        <button
          className={`filter-button ${filter === 'medium_risk' ? 'active' : ''}`}
          onClick={() => setFilter('medium_risk')}
        >
          Medium Risk
        </button>
        <button
          className={`filter-button ${filter === 'low_risk' ? 'active' : ''}`}
          onClick={() => setFilter('low_risk')}
        >
          Low Risk
        </button>
        <button
          className={`filter-button ${filter === 'no_risk' ? 'active' : ''}`}
          onClick={() => setFilter('no_risk')}
        >
          No Risk
        </button>
      </div>

      {/* Highlighted Clauses */}
      <div className="contract-text">
        {filteredClauses.map((clause) => (
          <span
            key={clause.id}
            className={`clause ${getRiskClass(clause.risk_label)}`}
            title={`${getRiskLabel(clause.risk_label)} (Confidence: ${(
              clause.confidence * 100
            ).toFixed(1)}%)`}
          >
            {clause.text}{' '}
          </span>
        ))}
      </div>
    </div>
  );
};

export default RiskVisualization;
