export function getApiBase(): string {
  const envBase = (import.meta as any)?.env?.VITE_API_BASE as string | undefined
  if (envBase && envBase.trim() !== '') {
    return envBase.replace(/\/+$/, '')
  }
  // In dev, rely on Vite proxy by using relative URLs
  if ((import.meta as any)?.env?.DEV) {
    return ''
  }
  const { protocol, host } = window.location
  return `${protocol}//${host}`
}

export function absoluteApiUrl(url?: string | null): string | null {
  if (!url) return null
  if (/^https?:\/\//i.test(url)) return url
  const base = getApiBase()
  if (!base) return url.startsWith('/') ? url : `/${url}`
  if (url.startsWith('/')) return `${base}${url}`
  return `${base}/${url}`
}


