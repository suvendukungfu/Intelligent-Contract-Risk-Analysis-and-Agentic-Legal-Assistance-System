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
        <p style={{ 
          background: 'var(--bg-tertiary)', 
          padding: '12px', 
          borderRadius: '6px',
          borderLeft: '2px solid var(--accent-primary)'
        }}>
          {report.contract_summary}
        </p>
      </div>

      {/* Overall Severity */}
      <div className="report-section">
        <h3>Overall Risk Assessment</h3>
        <span className={`severity-badge ${getSeverityClass(report.overall_severity)}`}>
          {report.overall_severity} Risk
        </span>
      </div>

      {/* Identified Risks */}
      <div className="report-section">
        <h3>Identified Risks ({report.identified_risks.length})</h3>
        {report.identified_risks.length === 0 ? (
          <p style={{ 
            color: 'var(--success)', 
            fontWeight: 500,
            padding: '16px',
            background: 'rgba(16, 185, 129, 0.1)',
            borderRadius: '8px',
            textAlign: 'center',
            border: '1px solid rgba(16, 185, 129, 0.3)'
          }}>
            No significant risks identified in this contract
          </p>
        ) : (
          report.identified_risks.map((risk, index) => (
            <div key={risk.clause_id} className="risk-card">
              <div>
                <span className={`severity-badge ${getSeverityClass(risk.severity)}`}>
                  {risk.severity} Risk
                </span>
              </div>

              <h4 style={{ marginTop: '8px', marginBottom: '12px' }}>
                {risk.risk_description}
              </h4>

              <div style={{ 
                marginTop: '12px',
                padding: '12px',
                background: 'var(--bg-primary)',
                borderRadius: '6px',
                borderLeft: '2px solid var(--border-secondary)'
              }}>
                <strong style={{ color: 'var(--text-tertiary)', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Clause</strong>
                <p style={{ 
                  fontStyle: 'italic', 
                  color: 'var(--text-secondary)', 
                  marginTop: '6px',
                  lineHeight: '1.5',
                  fontSize: '13px'
                }}>
                  "{risk.clause_text}"
                </p>
              </div>

              <div style={{ marginTop: '12px' }}>
                <strong style={{ color: 'var(--text-primary)', fontSize: '13px' }}>
                  Explanation
                </strong>
                <p style={{ marginTop: '6px' }}>{risk.explanation}</p>
              </div>

              <div style={{ marginTop: '12px' }}>
                <strong style={{ color: 'var(--text-primary)', fontSize: '13px' }}>
                  Potential Consequences
                </strong>
                <p style={{ marginTop: '6px' }}>{risk.consequences}</p>
              </div>

              {risk.mitigation_actions.length > 0 && (
                <div style={{ marginTop: '12px' }}>
                  <strong style={{ color: 'var(--text-primary)', fontSize: '13px' }}>
                    Mitigation Actions
                  </strong>
                  <ul className="mitigation-list">
                    {risk.mitigation_actions.map((action, idx) => (
                      <li key={idx}>{action}</li>
                    ))}
                  </ul>
                </div>
              )}

              {risk.legal_guidelines.length > 0 && (
                <div style={{ marginTop: '12px' }}>
                  <strong style={{ color: 'var(--text-primary)', fontSize: '13px' }}>
                    Legal Guidelines
                  </strong>
                  <ul className="mitigation-list">
                    {risk.legal_guidelines.map((guideline, idx) => (
                      <li key={idx} style={{ fontSize: '13px' }}>{guideline}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Legal Disclaimer */}
      <div className="legal-disclaimer">
        <strong>Legal Disclaimer:</strong> {report.legal_disclaimer}
      </div>
    </div>
  );
};

export default ReportView;
