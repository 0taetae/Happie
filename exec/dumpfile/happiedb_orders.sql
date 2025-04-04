-- MySQL dump 10.13  Distrib 8.0.38, for Win64 (x86_64)
--
-- Host: j12e103.p.ssafy.io    Database: happiedb
-- ------------------------------------------------------
-- Server version	8.0.41

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `orders`
--

DROP TABLE IF EXISTS `orders`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `orders` (
  `id` int NOT NULL AUTO_INCREMENT,
  `place` varchar(255) DEFAULT NULL,
  `robot` varchar(255) DEFAULT NULL,
  `state` varchar(255) DEFAULT NULL,
  `todo` varchar(255) DEFAULT NULL,
  `x` double DEFAULT NULL,
  `y` double DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=31 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `orders`
--

LOCK TABLES `orders` WRITE;
/*!40000 ALTER TABLE `orders` DISABLE KEYS */;
INSERT INTO `orders` VALUES (1,'501호실','robot1','진행 중','휠체어 전달',36.58,-52.52),(2,'502호실','robot2','진행 중','수리 중',36.65,-47.51),(3,'503호실','robot3','대기','출동',36.55,-42.56),(4,'501호실','robot3','진행 중','충전 중',36.58,-52.52),(5,'503호실','robot2','완료','휠체어 전달',36.55,-42.56),(6,'간호사실','robot2','완료','출동',55.37,-50.85),(7,'간호사실','robot2','완료','링거 전달',55.37,-50.85),(8,'503호실','robot2','완료','출동',36.55,-42.56),(9,'503호실','robot2','완료','휠체어 전달',36.55,-42.56),(10,'간호사실','robot2','완료','출동',55.37,-50.85),(11,'503호실','robot1','완료','방문',36.55,-42.56),(12,'병동','robot1','완료','운행',0,0),(13,'502호실','robot3','완료','휠체어 전달',36.65,-47.51),(14,'501호실','robot3','완료','링거 전달',36.58,-52.52),(15,'병동','robot1','대기','청소',0,0),(16,'502호실','robot1','진행 중','방문',36.65,-47.51),(17,'501호실','robot1','대기','링거 전달',36.58,-52.52),(19,'501호실','robot2','완료','링거 전달',36.58,-52.52),(20,'병동','robot2','완료','운행',0,0),(21,'503호실','robot2','완료','방문',36.55,-42.56),(22,'병동','robot2','대기','청소',0,0),(23,'간호사실','robot3','대기','출동',55.37,-50.85),(24,'병동','robot3','대기','운행',0,0),(25,'병동','robot3','대기','운행',0,0),(26,'502호실','robot3','대기','링거 전달',36.65,-47.51);
/*!40000 ALTER TABLE `orders` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-04-04 12:00:32
