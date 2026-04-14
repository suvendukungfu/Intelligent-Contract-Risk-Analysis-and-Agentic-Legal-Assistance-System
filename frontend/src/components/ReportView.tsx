import React from 'react';
import type { Milestone2Response } from '../api/client';

interface ReportViewProps {
  data: Milestone2Response;
}

const ReportView: React.FC<ReportViewProps> = ({ data }) => {
  const { report } = data;

  const getSeverityClass = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'high':
        return 'high';
      case 'medium':
        return 'medium';
      case 'low':
        return 'low';
      default:
        return '';
    }
  };

  return (
    <div className="report-view">
      <h2>Risk Analysis Report</h2>

      {/* Contract Summary */}
      <div className="report-section">
        <h3>Contract Summary</h3>
        <p>{report.contract_summary}</p>
      </div>

      {/* Overall Severity */}
      <div className="report-section">
        <h3>Overall Severity</h3>
        <span className={`severity-badge ${getSeverityClass(report.overall_severity)}`}>
          {report.overall_severity}
        </span>
      </div>

      {/* Identified Risks */}
      <div className="report-section">
        <h3>Identified Risks</h3>
        {report.identified_risks.map((risk, index) => (
          <div key={risk.clause_id} className="risk-card">
            <div>
              <span className={`severity-badge ${getSeverityClass(risk.severity)}`}>
                {risk.severity} Risk
              </span>
            </div>

            <h4>Risk {index + 1}: {risk.risk_description}</h4>

            <div style={{ marginTop: '10px' }}>
              <strong>Clause:</strong>
              <p style={{ fontStyle: 'italic', color: '#666', marginTop: '5px' }}>
                "{risk.clause_text}"
              </p>
            </div>

            <div style={{ marginTop: '10px' }}>
              <strong>Explanation:</strong>
              <p style={{ marginTop: '5px' }}>{risk.explanation}</p>
            </div>

            <div style={{ marginTop: '10px' }}>
              <strong>Potential Consequences:</strong>
              <p style={{ marginTop: '5px' }}>{risk.consequences}</p>
            </div>

            {risk.mitigation_actions.length > 0 && (
              <div style={{ marginTop: '10px' }}>
                <strong>Mitigation Actions:</strong>
                <ul className="mitigation-list">
                  {risk.mitigation_actions.map((action, idx) => (
                    <li key={idx}>{action}</li>
                  ))}
                </ul>
              </div>
            )}

            {risk.legal_guidelines.length > 0 && (
              <div style={{ marginTop: '10px' }}>
                <strong>Legal Guidelines:</strong>
                <ul className="mitigation-list">
                  {risk.legal_guidelines.map((guideline, idx) => (
                    <li key={idx}>{guideline}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Legal Disclaimer */}
      <div className="legal-disclaimer">
        <strong>Legal Disclaimer:</strong> {report.legal_disclaimer}
      </div>
    </div>
  );
};

export default ReportView;
