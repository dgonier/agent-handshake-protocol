output "instance_id" {
  value       = aws_instance.neo4j.id
  description = "EC2 instance id of the Neo4j box."
}

output "public_ip" {
  value       = aws_instance.neo4j.public_ip
  description = "Public IP for SSH and Bolt connections."
}

output "bolt_url" {
  value       = "bolt://${aws_instance.neo4j.public_ip}:7687"
  description = "Set as NEO4J_URI in your AHP environment."
}

output "browser_url" {
  value       = "http://${aws_instance.neo4j.public_ip}:7474"
  description = "Open in a browser to inspect / query the graph."
}

output "ssh_command" {
  value       = "ssh ubuntu@${aws_instance.neo4j.public_ip}"
  description = "Convenience: assumes your key is on the SSH agent."
}
