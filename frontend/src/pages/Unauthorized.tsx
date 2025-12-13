import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import './Unauthorized.css';

export default function Unauthorized() {
  const navigate = useNavigate();
  const { logout } = useAuth();

  const handleGoBack = () => {
    navigate(-1);
  };

  const handleLogout = async () => {
    await logout();
  };

  return (
    <div className="unauthorized-container">
      <div className="unauthorized-box">
        <h1>⛔ Unauthorized Access</h1>
        <p>You don't have permission to access this page.</p>
        <p>Please contact your administrator if you believe this is an error.</p>
        
        <div className="button-group">
          <button 
            onClick={handleGoBack} 
            className="btn btn-secondary"
          >
            Go Back
          </button>
          <button 
            onClick={handleLogout} 
            className="btn btn-primary"
          >
            Logout
          </button>
        </div>
      </div>
    </div>
  );
}
