(define (stream bedsidestocking3d)
  (:stream sample-place-pose
    :inputs (?o)
    :domain (Item ?o)
    :outputs (?p)
    :certified (Pose ?o ?p)
  )
  (:stream sample-grasp
    :inputs (?o)
    :domain (Item ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g)
  )
  (:stream inverse-kinematics
    :inputs (?o ?p ?g)
    :domain (and (Pose ?o ?p) (Grasp ?o ?g))
    :outputs (?q ?t)
    :certified (and (BConf ?q) (ATraj ?t) (Kin ?o ?p ?g ?q ?t))
  )

  (:stream test-usable
    :inputs (?o ?p)
    :domain (and (Requested ?o) (Pose ?o ?p))
    :certified (Usable ?o ?p)
  )
  (:stream test-clear
    :inputs (?o ?p ?o2 ?p2)
    :domain (and (Usable ?o ?p) (Pose ?o2 ?p2))
    :certified (Clear ?o ?p ?o2 ?p2)
  )
  (:stream test-cfree-pose-pose
    :inputs (?o1 ?p1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
    :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2)
  )
  (:stream test-cfree-traj-pose
    :inputs (?t ?o2 ?p2)
    :domain (and (ATraj ?t) (Pose ?o2 ?p2))
    :certified (CFreeTrajPose ?t ?o2 ?p2)
  )
)
