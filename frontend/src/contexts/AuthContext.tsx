import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { api } from '../services/api';
import { User } from '../types';
import Cookies from 'js-cookie';
import { useNavigate } from 'react-router-dom';

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  // Check authentication status on initial load
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = Cookies.get('auth_token');
        const storedUser = localStorage.getItem('user');
        
        if (token && storedUser) {
          try {
            // Set auth token in axios instance before making the request
            setAuthToken(token);
            
            // Verify token with backend
            const response = await api.get('/auth/me');
            setUser(response.data);
          } catch (error) {
            // Token is invalid, clear auth data
            console.error('Session expired or invalid token:', error);
            setAuthToken(null);
            Cookies.remove('auth_token', { path: '/' });
            localStorage.removeItem('user');
            navigate('/login');
          }
        } else {
          // No token or user data found, ensure we're logged out
          setAuthToken(null);
          setUser(null);
          navigate('/login');
        }
      } catch (error) {
        console.error('Auth check failed:', error);
        setAuthToken(null);
        Cookies.remove('auth_token', { path: '/' });
        localStorage.removeItem('user');
        navigate('/login');
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
    
    // Cleanup function to avoid memory leaks
    return () => {
      // Any cleanup if needed
    };
  }, []);

  // Set auth token in axios instance
  const setAuthToken = (token: string | null) => {
    if (token) {
      api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    } else {
      delete api.defaults.headers.common['Authorization'];
    }
  };

  const login = async (username: string, password: string) => {
    try {
      const response = await api.post('/auth/login', { username, password });
      const { user: userData, token } = response.data;
      
          // Set secure HTTP-only cookie
      const isProduction = import.meta.env.MODE === 'production';
      Cookies.set('auth_token', token, { 
        expires: 7, // 7 days
        secure: isProduction,
        sameSite: 'strict',
        path: '/'
      });
      
      // Set auth token in axios instance
      setAuthToken(token);
      
      // Store user data in local storage (non-sensitive data only)
      const { password: _, ...userWithoutPassword } = userData;
      localStorage.setItem('user', JSON.stringify(userWithoutPassword));
      setUser(userWithoutPassword);
      
      return userWithoutPassword;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Login failed');
    }
  };

  const logout = async () => {
    try {
      // Call logout API to invalidate token
      await api.post('/auth/logout');
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear auth data regardless of API call result
      setAuthToken(null);
      Cookies.remove('auth_token', { path: '/' });
      localStorage.removeItem('user');
      setUser(null);
      navigate('/login');
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        login,
        logout,
        loading,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

