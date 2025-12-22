import { UserRole } from '../types'

/**
 * Check if a user role has a specific permission.
 * Admin role automatically has ALL permissions.
 * 
 * @param userRole - The role of the user
 * @param requiredRole - The role required for the permission
 * @returns true if the user has the permission, false otherwise
 */
export function hasPermission(userRole: UserRole | undefined, requiredRole: UserRole | UserRole[]): boolean {
  if (!userRole) return false
  
  // Admin has all permissions
  if (userRole === 'admin') return true
  
  // Check if required role is an array
  if (Array.isArray(requiredRole)) {
    return requiredRole.includes(userRole)
  }
  
  // Check if user role matches required role
  return userRole === requiredRole
}

/**
 * Check if a user can access a specific feature.
 * Admin can access everything.
 */
export function canAccess(userRole: UserRole | undefined, feature: string): boolean {
  if (!userRole) return false
  
  // Admin can access everything
  if (userRole === 'admin') return true
  
  // Define feature permissions for each role
  const permissions: Record<string, UserRole[]> = {
    'upload': ['user', 'field', 'admin'],
    'gallery': ['user', 'field', 'onsite', 'admin'],
    'edit': ['onsite', 'admin'],
    'verify': ['onsite', 'admin'],
    'user-management': ['admin'],
    'audit-logs': ['admin'],
  }
  
  const allowedRoles = permissions[feature] || []
  return allowedRoles.includes(userRole)
}

/**
 * Get all roles that have a specific permission.
 */
export function getRolesWithPermission(requiredRole: UserRole | UserRole[]): UserRole[] {
  const roles: UserRole[] = ['admin', 'user', 'field', 'onsite']
  
  if (Array.isArray(requiredRole)) {
    return ['admin', ...requiredRole]
  }
  
  return ['admin', requiredRole]
}

