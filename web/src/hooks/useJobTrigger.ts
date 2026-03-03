import { useState, useEffect, useCallback, useRef } from "react"
import { useQueryClient, useQuery, useMutation } from "@tanstack/react-query"
import { api, type JobState } from "@/lib/api"

type JobPhase = "idle" | "running" | "completed" | "failed"

interface UseJobTriggerResult {
  trigger: () => void
  phase: JobPhase
  error: string | null
}

/**
 * Reusable hook for triggering a background job, polling for completion,
 * and auto-invalidating query keys when done.
 *
 * @param jobName - The job endpoint name ("momentum", "pead", "strategy-compare")
 * @param queryKeysToInvalidate - TanStack Query keys to invalidate on completion
 */
export function useJobTrigger(
  jobName: string,
  queryKeysToInvalidate: string[][],
): UseJobTriggerResult {
  const queryClient = useQueryClient()
  const [phase, setPhase] = useState<JobPhase>("idle")
  const [error, setError] = useState<string | null>(null)
  const jobIdRef = useRef<string | null>(null)

  // Poll job status only while running
  const { data: statusData } = useQuery({
    queryKey: ["job-status", jobName],
    queryFn: api.jobStatus,
    enabled: phase === "running",
    refetchInterval: 2000,
  })

  // Watch for our job's completion in the polled status
  useEffect(() => {
    if (phase !== "running" || !statusData?.jobs || !jobIdRef.current) return

    const job = statusData.jobs.find((j: JobState) => j.id === jobIdRef.current)
    if (!job) return

    if (job.status === "completed") {
      setPhase("completed")
      jobIdRef.current = null
      // Invalidate cached queries so pages auto-refetch fresh data
      for (const key of queryKeysToInvalidate) {
        queryClient.invalidateQueries({ queryKey: key })
      }
      // Reset to idle after a brief flash
      setTimeout(() => setPhase("idle"), 2000)
    } else if (job.status === "failed") {
      setPhase("failed")
      setError(job.error ?? "Job failed")
      jobIdRef.current = null
      setTimeout(() => {
        setPhase("idle")
        setError(null)
      }, 5000)
    }
  }, [statusData, phase, queryClient, queryKeysToInvalidate])

  const mutation = useMutation({
    mutationFn: () => api.triggerJob(jobName),
    onSuccess: (data) => {
      jobIdRef.current = data.job_id
      setPhase("running")
      setError(null)
    },
    onError: (err: Error) => {
      // 409 = job already running — start polling for it
      if (err.message.startsWith("409")) {
        setPhase("running")
        setError(null)
      } else {
        setPhase("failed")
        setError(err.message)
        setTimeout(() => {
          setPhase("idle")
          setError(null)
        }, 5000)
      }
    },
  })

  const trigger = useCallback(() => {
    if (phase === "running") return
    mutation.mutate()
  }, [phase, mutation])

  return { trigger, phase, error }
}
