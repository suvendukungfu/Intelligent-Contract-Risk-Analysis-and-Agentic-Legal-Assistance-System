import React from 'react';
import type { Milestone1Response } from '../api/client';

interface RiskVisualizationProps {
  data: Milestone1Response;
}

const RiskVisualization: React.FC<RiskVisualizationProps> = ({ data }) => {
  return (
    <div className="risk-visualization">
      <h2>Risk Classification Results</h2>

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

      {/* Separated Risk Sections */}
      <div>
        {/* High Risk Clauses */}
        {data.clauses.filter(c => c.risk_label === 'high_risk').length > 0 && (
          <div style={{ marginBottom: '24px' }}>
            <p style={{ 
              fontSize: '13px', 
              color: '#ef4444', 
              marginBottom: '8px',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.05em'
            }}>
              High Risk Clauses ({data.clauses.filter(c => c.risk_label === 'high_risk').length})
            </p>
            <div className="contract-text">
              {data.clauses
                .filter(c => c.risk_label === 'high_risk')
                .map((clause) => (
                  <span
                    key={clause.id}
                    className="clause high-risk"
                    title={`High Risk (Confidence: ${(clause.confidence * 100).toFixed(1)}%)`}
                  >
                    {clause.text}{' '}
                  </span>
                ))}
            </div>
          </div>
        )}

        {/* Medium Risk Clauses */}
        {data.clauses.filter(c => c.risk_label === 'medium_risk').length > 0 && (
          <div style={{ marginBottom: '24px' }}>
            <p style={{ 
              fontSize: '13px', 
              color: '#f59e0b', 
              marginBottom: '8px',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.05em'
            }}>
              Medium Risk Clauses ({data.clauses.filter(c => c.risk_label === 'medium_risk').length})
            </p>
            <div className="contract-text">
              {data.clauses
                .filter(c => c.risk_label === 'medium_risk')
                .map((clause) => (
                  <span
                    key={clause.id}
                    className="clause medium-risk"
                    title={`Medium Risk (Confidence: ${(clause.confidence * 100).toFixed(1)}%)`}
                  >
                    {clause.text}{' '}
                  </span>
                ))}
            </div>
          </div>
        )}

        {/* Low Risk Clauses */}
        {data.clauses.filter(c => c.risk_label === 'low_risk').length > 0 && (
          <div style={{ marginBottom: '24px' }}>
            <p style={{ 
              fontSize: '13px', 
              color: '#eab308', 
              marginBottom: '8px',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.05em'
            }}>
              Low Risk Clauses ({data.clauses.filter(c => c.risk_label === 'low_risk').length})
            </p>
            <div className="contract-text">
              {data.clauses
                .filter(c => c.risk_label === 'low_risk')
                .map((clause) => (
                  <span
                    key={clause.id}
                    className="clause low-risk"
                    title={`Low Risk (Confidence: ${(clause.confidence * 100).toFixed(1)}%)`}
                  >
                    {clause.text}{' '}
                  </span>
                ))}
            </div>
          </div>
        )}

        {/* No Risk Clauses */}
        {data.clauses.filter(c => c.risk_label === 'no_risk').length > 0 && (
          <div style={{ marginBottom: '24px' }}>
            <p style={{ 
              fontSize: '13px', 
              color: '#10b981', 
              marginBottom: '8px',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.05em'
            }}>
              No Risk Clauses ({data.clauses.filter(c => c.risk_label === 'no_risk').length})
            </p>
            <div className="contract-text">
              {data.clauses
                .filter(c => c.risk_label === 'no_risk')
                .map((clause) => (
                  <span
                    key={clause.id}
                    className="clause no-risk"
                    title={`No Risk (Confidence: ${(clause.confidence * 100).toFixed(1)}%)`}
                  >
                    {clause.text}{' '}
                  </span>
                ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default RiskVisualization;
